//! Explicit `DuckDuckGo` HTML search provider for the CLI.

use std::time::Duration;

use async_trait::async_trait;
use emelex::agent::{AgentCancellation, WebSearchError, WebSearchProvider, WebSearchResult};
use futures::StreamExt as _;
use url::Url;

const ENDPOINT: &str = "https://html.duckduckgo.com/html/";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_TITLE_BYTES: usize = 512;
const MAX_URL_BYTES: usize = 2_048;
const MAX_SNIPPET_BYTES: usize = 4_096;

/// Bounded, credential-free search against `DuckDuckGo`'s no-JavaScript page.
pub(crate) struct DuckDuckGoSearch {
	client: reqwest::Client,
}

impl DuckDuckGoSearch {
	/// Construct the fixed-policy search client.
	pub(crate) fn new() -> Result<Self, reqwest::Error> {
		let client = reqwest::Client::builder()
			.connect_timeout(CONNECT_TIMEOUT)
			.timeout(SEARCH_TIMEOUT)
			.redirect(reqwest::redirect::Policy::none())
			.no_proxy()
			.user_agent(concat!("emelex/", env!("CARGO_PKG_VERSION")))
			.build()?;
		Ok(Self { client })
	}
}

#[async_trait]
impl WebSearchProvider for DuckDuckGoSearch {
	fn implementation_identity(&self) -> String {
		"emelex.cli.duckduckgo-html@1".to_string()
	}

	async fn search(
		&self,
		query: &str,
		limit: usize,
		cancellation: &AgentCancellation,
	) -> Result<Vec<WebSearchResult>, WebSearchError> {
		let response = tokio::select! {
			biased;
			() = cancellation.cancelled() => {
				return Err(WebSearchError::new("web search cancelled"));
			}
			response = self.client.get(ENDPOINT).query(&[("q", query)]).send() => {
				response.map_err(|error| WebSearchError::new(format!("request failed: {error}")))?
			}
		};
		if !response.status().is_success() {
			return Err(WebSearchError::new(format!(
				"DuckDuckGo returned HTTP {}",
				response.status()
			)));
		}

		let mut body = Vec::new();
		let mut stream = response.bytes_stream();
		loop {
			let next = tokio::select! {
				biased;
				() = cancellation.cancelled() => {
					return Err(WebSearchError::new("web search cancelled"));
				}
				next = stream.next() => next,
			};
			let Some(chunk) = next else {
				break;
			};
			let chunk = chunk
				.map_err(|error| WebSearchError::new(format!("response read failed: {error}")))?;
			if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
				return Err(WebSearchError::new(format!(
					"DuckDuckGo response exceeds {MAX_RESPONSE_BYTES} bytes"
				)));
			}
			body.extend_from_slice(&chunk);
		}
		if cancellation.is_cancelled() {
			return Err(WebSearchError::new("web search cancelled"));
		}
		let html = std::str::from_utf8(&body)
			.map_err(|_| WebSearchError::new("DuckDuckGo returned non-UTF-8 HTML"))?;
		if is_interactive_challenge(html) {
			return Err(WebSearchError::new(
				"DuckDuckGo requires an interactive challenge; web search is unavailable",
			));
		}
		Ok(parse_results(html, limit))
	}
}

fn is_interactive_challenge(html: &str) -> bool {
	let lowercase = html.to_ascii_lowercase();
	lowercase.contains("challenge-form") || lowercase.contains("anomaly-modal__title")
}

/// Parse `DuckDuckGo`'s current server-rendered result markup.
///
/// Markup is not an API contract. Unknown or interstitial pages therefore
/// produce no results instead of manufacturing URLs.
fn parse_results(html: &str, limit: usize) -> Vec<WebSearchResult> {
	let mut results = Vec::new();
	let mut sections = html.split("result__a");
	let mut previous = sections.next();
	for section in sections {
		if results.len() >= limit {
			break;
		}
		let before_tail =
			previous.map(|before| before.rfind('<').map_or(before, |open| &before[open..]));
		let after_head = section.find('>').map_or(section, |end| &section[..end]);
		let href = before_tail
			.and_then(|tag| attribute(tag, "href"))
			.or_else(|| attribute(after_head, "href"));
		previous = Some(section);

		let Some(url) = href.and_then(resolve_result_url) else {
			continue;
		};
		let title = section
			.find('>')
			.and_then(|start| {
				section[start + 1..]
					.find("</a>")
					.map(|end| &section[start + 1..start + 1 + end])
			})
			.map(|value| bounded_plain_text(value, MAX_TITLE_BYTES))
			.unwrap_or_default();
		if title.is_empty() {
			continue;
		}
		let snippet = section
			.find("result__snippet")
			.and_then(|at| {
				let trailing = &section[at..];
				let start = trailing.find('>')?;
				let end = trailing[start + 1..].find("</a>")?;
				Some(&trailing[start + 1..start + 1 + end])
			})
			.map(|value| bounded_plain_text(value, MAX_SNIPPET_BYTES))
			.unwrap_or_default();
		results.push(WebSearchResult::new(title, url, snippet));
	}
	results
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
	let needle = format!("{name}=\"");
	tag.rfind(&needle)
		.map(|at| &tag[at + needle.len()..])
		.and_then(|value| value.split('"').next())
}

fn resolve_result_url(href: &str) -> Option<String> {
	let decoded = decode_entities(href);
	let absolute = if decoded.starts_with("//") {
		format!("https:{decoded}")
	} else if decoded.starts_with('/') {
		format!("https://duckduckgo.com{decoded}")
	} else {
		decoded
	};
	let mut url = Url::parse(&absolute).ok()?;
	if url.host_str() == Some("duckduckgo.com") && url.path() == "/l/" {
		let target = url
			.query_pairs()
			.find_map(|(name, value)| (name == "uddg").then(|| value.into_owned()))?;
		url = Url::parse(&target).ok()?;
	}
	if !matches!(url.scheme(), "http" | "https")
		|| !url.username().is_empty()
		|| url.password().is_some()
		|| url.as_str().len() > MAX_URL_BYTES
	{
		return None;
	}
	Some(url.into())
}

fn bounded_plain_text(html: &str, maximum: usize) -> String {
	let mut plain = String::with_capacity(html.len().min(maximum));
	let mut in_tag = false;
	for character in html.chars() {
		match character {
			'<' => in_tag = true,
			'>' if in_tag => in_tag = false,
			_ if !in_tag => plain.push(character),
			_ => {}
		}
	}
	let decoded = decode_entities(&plain);
	let collapsed = decoded.split_whitespace().collect::<Vec<_>>().join(" ");
	if collapsed.len() <= maximum {
		return collapsed;
	}
	let mut end = maximum;
	while !collapsed.is_char_boundary(end) {
		end -= 1;
	}
	collapsed[..end].trim_end().to_string()
}

fn decode_entities(value: &str) -> String {
	value
		.replace("&amp;", "&")
		.replace("&quot;", "\"")
		.replace("&#39;", "'")
		.replace("&#x27;", "'")
		.replace("&apos;", "'")
		.replace("&lt;", "<")
		.replace("&gt;", ">")
		.replace("&nbsp;", " ")
}

#[cfg(test)]
mod tests {
	use super::*;

	const FIXTURE: &str = r#"
		<a class="result__a" href="https://www.rust-lang.org/">Rust <b>Language</b></a>
		<a class="result__snippet">Fast &amp; reliable.</a>
		<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F&amp;rut=x" class="result__a">The Book</a>
		<a class="result__snippet">Learn &quot;Rust&quot;.</a>
		<a class="result__a" href="javascript:alert(1)">Unsafe</a>
		<a class="result__snippet">Ignored.</a>
	"#;

	#[test]
	fn parses_direct_and_wrapped_http_results() {
		let results = parse_results(FIXTURE, 8);
		assert_eq!(results.len(), 2);
		assert_eq!(results[0].title, "Rust Language");
		assert_eq!(results[0].url, "https://www.rust-lang.org/");
		assert_eq!(results[0].snippet, "Fast & reliable.");
		assert_eq!(results[1].url, "https://doc.rust-lang.org/book/");
		assert_eq!(results[1].snippet, "Learn \"Rust\".");
	}

	#[test]
	fn caps_results_and_utf8_fields() {
		assert_eq!(parse_results(FIXTURE, 1).len(), 1);
		let long = format!("<b>{}</b>", "🙂".repeat(MAX_TITLE_BYTES));
		let bounded = bounded_plain_text(&long, MAX_TITLE_BYTES);
		assert!(bounded.len() <= MAX_TITLE_BYTES);
		assert!(bounded.is_char_boundary(bounded.len()));
	}

	#[test]
	fn interstitial_or_unsupported_urls_produce_no_results() {
		assert!(parse_results("<html>rate limited</html>", 8).is_empty());
		assert!(is_interactive_challenge(
			"<FORM ID=\"challenge-form\">challenge</FORM>"
		));
		assert!(is_interactive_challenge(
			"<div class=\"anomaly-modal__title\">Bots</div>"
		));
		assert!(!is_interactive_challenge(FIXTURE));
		assert!(resolve_result_url("file:///etc/passwd").is_none());
		assert!(resolve_result_url("javascript:alert(1)").is_none());
		assert!(resolve_result_url("https://user:secret@example.com/").is_none());
	}
}
