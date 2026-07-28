//! Retrieval-augmented generation on a local MLX model, fully offline:
//! rig's in-memory vector store + `dynamic_context` + emelex.
//!
//! The embedder here is a deliberately tiny, deterministic character
//! trigram hasher - no network, no second model - good enough to rank
//! runbook snippets by lexical overlap so the example is
//! self-contained. Lexical means literal: a query about a database that
//! "died" will NOT retrieve a "failover" doc; closing that semantic gap
//! is exactly what a real embedding provider buys you in production. On
//! each prompt, rig retrieves the top matches and injects them as
//! documents, which emelex renders into the chat template.
//!
//! ```sh
//! cargo run -p emelex --release --example rag -- \
//!   "$EMELEX_TEST_MODEL"
//! ```

// Example code: unwraps and panics are acceptable here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use emelex::ReasoningExt as _;
use rig_core::{
	OneOrMany,
	client::Nothing,
	completion::Prompt,
	embeddings::{Embedding, EmbeddingError, EmbeddingModel},
	vector_store::{
		VectorStoreIndex, in_memory_store::InMemoryVectorStore, request::VectorSearchRequest,
	},
};

/// Deterministic character-trigram embedder: every 3-gram of every
/// lowercased word hashes to one of 256 buckets; the vector is the
/// normalized bucket histogram.
#[derive(Clone)]
struct HashEmbedder;

const DIMS: usize = 256;

fn embed_one(text: &str) -> Vec<f64> {
	let mut v = vec![0.0f64; DIMS];
	let cleaned: String = text
		.to_lowercase()
		.chars()
		.map(|c| if c.is_alphanumeric() { c } else { ' ' })
		.collect();
	for word in cleaned.split_whitespace() {
		let bytes = word.as_bytes();
		for i in 0..bytes.len().saturating_sub(2).max(1).min(bytes.len()) {
			let gram = &bytes[i..(i + 3).min(bytes.len())];
			let mut hash = 0u64;
			for &b in gram {
				hash = hash.wrapping_mul(31).wrapping_add(u64::from(b));
			}
			v[(hash % DIMS as u64) as usize] += 1.0;
		}
	}
	let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
	v.iter().map(|x| x / norm).collect()
}

impl EmbeddingModel for HashEmbedder {
	type Client = Nothing;

	const MAX_DOCUMENTS: usize = 64;

	fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
		Self
	}

	fn ndims(&self) -> usize {
		DIMS
	}

	async fn embed_texts(
		&self,
		texts: impl IntoIterator<Item = String> + Send,
	) -> Result<Vec<Embedding>, EmbeddingError> {
		Ok(texts
			.into_iter()
			.map(|document| {
				let vec = embed_one(&document);
				Embedding { document, vec }
			})
			.collect())
	}
}

const RUNBOOK: &[&str] = &[
	"TLS certificates: rotate with `certctl rotate --service <name>`; \
	 certificates auto-renew 30 days before expiry, but a stuck renewal \
	 requires deleting the order in the certctl dashboard first.",
	"Database failover: promote the eu-west-1 replica with `pgctl promote`; \
	 expect up to 40 seconds of write unavailability and always announce in \
	 #incident before promoting.",
	"Deploy rollback: `shipit rollback <service>` reverts to the last \
	 known-good release; rollbacks skip canary and take about 2 minutes.",
	"On-call handover: rotations switch Mondays 10:00; the outgoing engineer \
	 posts a handover note covering open alerts and silenced monitors.",
	"Cache invalidation: the edge cache honors `purge-key` headers; a full \
	 flush needs approval from the platform lead and takes 10 minutes to \
	 propagate.",
	"Billing reconciliation: the nightly job compares ledger entries with \
	 gateway settlements; mismatches above 100 EUR page the on-call immediately.",
];

const QUESTIONS: &[&str] = &[
	"A TLS certificate renewal seems stuck - how do I fix it?",
	"The primary database is down - how do I fail over to the replica?",
];

fn store_with_docs(embeddings: &[Embedding]) -> InMemoryVectorStore<String> {
	InMemoryVectorStore::from_documents(
		RUNBOOK
			.iter()
			.zip(embeddings)
			.map(|(doc, e)| ((*doc).to_string(), OneOrMany::one(e.clone()))),
	)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let model_dir = std::env::args().nth(1).expect("usage: rag <mlx-model-dir>");

	let embeddings = HashEmbedder
		.embed_texts(RUNBOOK.iter().map(|s| (*s).to_string()))
		.await?;

	// One index for peeking at what retrieval selects, one moved into
	// the agent for automatic per-prompt injection.
	let peek_index = store_with_docs(&embeddings).index(HashEmbedder);
	let agent_index = store_with_docs(&embeddings).index(HashEmbedder);

	let agent = emelex::Client::from_path(model_dir)?
		.agent()
		.preamble(
			"You are an SRE assistant. Answer ONLY from the provided runbook \
			 documents, quoting the exact commands. If the documents don't cover \
			 it, say so.",
		)
		.enable_thinking(false)
		.dynamic_context(2, agent_index)
		.build();

	for question in QUESTIONS {
		println!("== {question}");
		let request = VectorSearchRequest::builder()
			.query(*question)
			.samples(2)
			.build();
		for (score, _id, doc) in peek_index.top_n::<String>(request).await? {
			let head = doc.chars().take(64).collect::<String>();
			println!("  [retrieved {score:.3}] {head}...");
		}
		let answer = agent.prompt(*question).await?;
		println!("{answer}\n");
	}

	Ok(())
}
