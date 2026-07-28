//! Explicit, bounded network tools and local datetime utility.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{FixedOffset, SecondsFormat, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use url::Url;

use super::{
	AgentTool, ApprovalRequirement, BoundedJsonError, ToolContext, ToolError, ToolOutput,
	serialize_json_pretty_bounded,
};
use crate::generation::ToolDefinition;

/// Hard ceiling accepted by [`web_fetch_tool_with_limit`] for one response.
pub const MAX_WEB_RESPONSE_BYTES: usize = 1024 * 1024;
const DEFAULT_FETCH_BYTES: usize = 512 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_SEARCH_RESULTS: usize = 10;
const DEFAULT_SEARCH_RESULTS: usize = 5;
const MAX_SEARCH_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_SEARCH_QUERY_BYTES: usize = 4096;
const MAX_RESULT_TITLE_BYTES: usize = 512;
const MAX_RESULT_URL_BYTES: usize = 2048;
const MAX_RESULT_SNIPPET_BYTES: usize = 4096;

/// Web-tool construction failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WebError {
	/// Requested response ceiling is zero or exceeds the hard bound.
	#[error("web response bytes must be in 1..={maximum}, got {requested}")]
	ResponseLimit {
		/// Requested ceiling.
		requested: usize,
		/// Hard ceiling.
		maximum: usize,
	},
	/// HTTP client policy could not be constructed.
	#[error("cannot construct bounded web client: {0}")]
	Client(#[source] reqwest::Error),
}

/// Provider-owned web-search failure.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct WebSearchError {
	message: String,
}

impl WebSearchError {
	/// Construct a provider diagnostic.
	pub fn new(message: impl Into<String>) -> Self {
		Self {
			message: message.into(),
		}
	}
}

/// One provider-neutral web search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WebSearchResult {
	/// Human-readable result title.
	pub title: String,
	/// Absolute HTTP(S) target.
	pub url: String,
	/// Short provider-supplied excerpt.
	pub snippet: String,
}

impl WebSearchResult {
	/// Construct one provider result.
	#[must_use]
	pub fn new(
		title: impl Into<String>,
		url: impl Into<String>,
		snippet: impl Into<String>,
	) -> Self {
		Self {
			title: title.into(),
			url: url.into(),
			snippet: snippet.into(),
		}
	}
}

/// Explicit backend for generic `web_search`.
///
/// Emelex deliberately ships no hidden search vendor. Applications opt in by
/// providing this trait and are responsible for their backend's credentials
/// and privacy policy.
#[async_trait]
pub trait WebSearchProvider: Send + Sync {
	/// Stable backend/config identity used by durable agent sessions.
	///
	/// Override when backend behavior or configuration can change
	/// independently of the concrete Rust type or Emelex package version.
	fn implementation_identity(&self) -> String {
		format!("rust:{}@protocol-1", std::any::type_name::<Self>())
	}

	/// Search for at most `limit` results.
	///
	/// Emelex may drop this future as soon as `cancellation` fires. The
	/// implementation must not detach I/O or host effects that outlive that
	/// drop. Any internally spawned work must observe the supplied
	/// cancellation and finish before the future returns or is dropped.
	///
	/// # Errors
	///
	/// Returns a provider-owned, display-safe diagnostic.
	async fn search(
		&self,
		query: &str,
		limit: usize,
		cancellation: &super::AgentCancellation,
	) -> Result<Vec<WebSearchResult>, WebSearchError>;
}

struct WebFetchTool {
	client: reqwest::Client,
	max_response_bytes: usize,
}

/// Construct opt-in bounded `web_fetch`.
///
/// # Errors
///
/// Returns an HTTP-client policy construction error.
pub fn web_fetch_tool() -> Result<Arc<dyn AgentTool>, WebError> {
	web_fetch_tool_with_limit(DEFAULT_FETCH_BYTES)
}

/// Construct opt-in `web_fetch` with an authoritative response ceiling.
///
/// # Errors
///
/// Returns a response-limit or HTTP-client policy construction error.
pub fn web_fetch_tool_with_limit(
	max_response_bytes: usize,
) -> Result<Arc<dyn AgentTool>, WebError> {
	if !(1..=MAX_WEB_RESPONSE_BYTES).contains(&max_response_bytes) {
		return Err(WebError::ResponseLimit {
			requested: max_response_bytes,
			maximum: MAX_WEB_RESPONSE_BYTES,
		});
	}
	let client = reqwest::Client::builder()
		.connect_timeout(CONNECT_TIMEOUT)
		.timeout(FETCH_TIMEOUT)
		// Redirects cross the exact-URL approval boundary. A 3xx response is
		// returned to the model so the destination requires a new tool call and
		// therefore a new one-shot approval.
		.redirect(reqwest::redirect::Policy::none())
		.no_proxy()
		.user_agent(concat!("emelex/", env!("CARGO_PKG_VERSION")))
		.build()
		.map_err(WebError::Client)?;
	Ok(Arc::new(WebFetchTool {
		client,
		max_response_bytes,
	}))
}

#[async_trait]
impl AgentTool for WebFetchTool {
	fn implementation_identity(&self) -> String {
		format!(
			"emelex.web_fetch@1;max_response_bytes={}",
			self.max_response_bytes
		)
	}

	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"web_fetch",
			"Fetch one HTTP(S) resource without cookies, credentials, or ambient proxy settings.",
			serde_json::json!({
				"type": "object",
				"properties": {
					"url": {
						"type": "string",
						"minLength": 1,
						"maxLength": MAX_RESULT_URL_BYTES
					},
						"max_bytes": {
							"type": "integer",
							"minimum": 1,
							"maximum": self.max_response_bytes
					}
				},
				"required": ["url"],
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		let target = arguments
			.get("url")
			.and_then(serde_json::Value::as_str)
			.and_then(|input| validate_http_url(input).ok())
			.map_or_else(
				|| "<invalid model-provided URL>".to_string(),
				|mut url| {
					url.set_fragment(None);
					format!("{:?}", url.as_str())
				},
			);
		ApprovalRequirement::Required {
			reason: format!("network request to {target}"),
		}
	}

	#[expect(
		clippy::too_many_lines,
		reason = "one linear pipeline keeps URL approval, redirect, and body bounds auditable"
	)]
	async fn invoke(
		&self,
		context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		if !context.approved() {
			return Err(ToolError::Fatal(
				"web_fetch reached execution without approval".to_string(),
			));
		}
		let args: FetchArgs = parse_args(arguments)?;
		let max_bytes = args.max_bytes.unwrap_or(self.max_response_bytes);
		if !(1..=self.max_response_bytes).contains(&max_bytes) {
			return Err(respond(format!(
				"max_bytes must be in 1..={}",
				self.max_response_bytes
			)));
		}
		let mut url = validate_http_url(&args.url)?;
		url.set_fragment(None);
		let request = self
			.client
			.get(url.clone())
			.build()
			.map_err(|error| respond(format!("cannot construct request for {url}: {error}")))?;
		let response = tokio::select! {
			biased;
			() = context.cancellation().cancelled() => {
				return Err(ToolError::Cancelled);
			}
			response = self.client.execute(request) => response,
		}
		.map_err(|error| respond(format!("request to {url} failed: {error}")))?;
		if let Some(target) = redirect_target(&url, &response)? {
			return Ok(ToolOutput::error(format!(
				"redirect to {target:?} requires a new web_fetch call and one-shot approval"
			)));
		}
		if response
			.content_length()
			.is_some_and(|length| length > max_bytes as u64)
		{
			return Err(respond(format!(
				"response Content-Length exceeds {max_bytes} bytes"
			)));
		}
		let status = response.status();
		let final_url = response.url().to_string();
		if final_url.len() > MAX_RESULT_URL_BYTES {
			return Err(respond("redirected URL exceeds output bound"));
		}
		let content_type = response
			.headers()
			.get(reqwest::header::CONTENT_TYPE)
			.and_then(|value| value.to_str().ok())
			.unwrap_or("application/octet-stream")
			.to_string();
		if content_type.len() > 512 {
			return Err(respond("response Content-Type exceeds output bound"));
		}
		let mut body = Vec::new();
		let mut stream = response.bytes_stream();
		loop {
			let chunk = tokio::select! {
				biased;
				() = context.cancellation().cancelled() => {
					return Err(ToolError::Cancelled);
				}
				chunk = stream.next() => chunk,
			};
			let Some(chunk) = chunk else {
				break;
			};
			let chunk = chunk.map_err(|error| respond(format!("response body failed: {error}")))?;
			let Some(total) = body.len().checked_add(chunk.len()) else {
				return Err(respond("response size overflow"));
			};
			if total > max_bytes {
				return Err(respond(format!("response body exceeds {max_bytes} bytes")));
			}
			body.extend_from_slice(&chunk);
		}
		let output = FetchOutput {
			url: final_url,
			status: status.as_u16(),
			content_type,
			body: String::from_utf8_lossy(&body).into_owned(),
		};
		let content = match serialize_json_pretty_bounded(&output, max_bytes) {
			Ok(content) => content,
			Err(BoundedJsonError::Limit { .. }) => {
				return Err(respond(format!(
					"serialized response exceeds {max_bytes} bytes"
				)));
			}
			Err(error @ (BoundedJsonError::Serialize(_) | BoundedJsonError::Utf8)) => {
				return Err(ToolError::Fatal(format!(
					"cannot serialize web response: {error}"
				)));
			}
		};
		Ok(if status.is_success() {
			ToolOutput::success(content)
		} else {
			ToolOutput::error(content)
		})
	}
}

fn redirect_target(
	request_url: &Url,
	response: &reqwest::Response,
) -> Result<Option<String>, ToolError> {
	if !response.status().is_redirection() {
		return Ok(None);
	}
	let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
		return Ok(None);
	};
	let location = location
		.to_str()
		.map_err(|_| respond("redirect Location is not valid ASCII/UTF-8"))?;
	if location.len() > MAX_RESULT_URL_BYTES {
		return Err(respond("redirect Location exceeds URL bound"));
	}
	let mut target = request_url
		.join(location)
		.map_err(|error| respond(format!("invalid redirect Location: {error}")))?;
	target = validate_http_url(target.as_str())?;
	target.set_fragment(None);
	let target = target.to_string();
	if target.len() > MAX_RESULT_URL_BYTES {
		return Err(respond("redirect target exceeds URL bound"));
	}
	Ok(Some(target))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs {
	url: String,
	#[serde(default)]
	max_bytes: Option<usize>,
}

#[derive(Serialize)]
struct FetchOutput {
	url: String,
	status: u16,
	content_type: String,
	body: String,
}

struct WebSearchTool {
	provider: Arc<dyn WebSearchProvider>,
}

/// Construct generic `web_search` from an explicit backend.
pub fn web_search_tool(provider: Arc<dyn WebSearchProvider>) -> Arc<dyn AgentTool> {
	Arc::new(WebSearchTool { provider })
}

#[async_trait]
impl AgentTool for WebSearchTool {
	fn implementation_identity(&self) -> String {
		format!(
			"emelex.web_search@1;provider={}",
			self.provider.implementation_identity()
		)
	}

	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"web_search",
			"Search the web through the application's explicitly configured provider.",
			serde_json::json!({
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"minLength": 1,
						"maxLength": MAX_SEARCH_QUERY_BYTES
					},
					"limit": {
						"type": "integer",
						"minimum": 1,
						"maximum": MAX_SEARCH_RESULTS
					}
				},
				"required": ["query"],
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		ApprovalRequirement::Required {
			reason: "network request through configured web-search provider".to_string(),
		}
	}

	async fn invoke(
		&self,
		context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		if !context.approved() {
			return Err(ToolError::Fatal(
				"web_search reached execution without approval".to_string(),
			));
		}
		let args: SearchArgs = parse_args(arguments)?;
		if args.query.is_empty() || args.query.len() > MAX_SEARCH_QUERY_BYTES {
			return Err(respond(format!(
				"query must be in 1..={MAX_SEARCH_QUERY_BYTES} bytes"
			)));
		}
		if !(1..=MAX_SEARCH_RESULTS).contains(&args.limit) {
			return Err(respond(format!(
				"limit must be in 1..={MAX_SEARCH_RESULTS}"
			)));
		}
		let results = tokio::select! {
			biased;
			() = context.cancellation().cancelled() => {
				return Err(ToolError::Cancelled);
			}
			results = self.provider.search(
				&args.query,
				args.limit,
				context.cancellation(),
			) => results,
		}
		.map_err(|error| respond(format!("web search failed: {error}")))?;
		if results.len() > args.limit {
			return Err(ToolError::Fatal(format!(
				"web-search provider returned {} results for limit {}",
				results.len(),
				args.limit
			)));
		}
		validate_search_results(&results)?;
		let output =
			serialize_json_pretty_bounded(&results, MAX_SEARCH_OUTPUT_BYTES).map_err(|error| {
				ToolError::Fatal(format!("cannot serialize bounded search results: {error}"))
			})?;
		Ok(ToolOutput::success(output))
	}
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
	query: String,
	#[serde(default = "default_search_results")]
	limit: usize,
}

const fn default_search_results() -> usize {
	DEFAULT_SEARCH_RESULTS
}

fn validate_search_results(results: &[WebSearchResult]) -> Result<(), ToolError> {
	for result in results {
		if result.title.len() > MAX_RESULT_TITLE_BYTES {
			return Err(ToolError::Fatal(
				"web-search result title exceeds bound".to_string(),
			));
		}
		if result.url.len() > MAX_RESULT_URL_BYTES {
			return Err(ToolError::Fatal(
				"web-search result URL exceeds bound".to_string(),
			));
		}
		if result.snippet.len() > MAX_RESULT_SNIPPET_BYTES {
			return Err(ToolError::Fatal(
				"web-search result snippet exceeds bound".to_string(),
			));
		}
		validate_http_url(&result.url).map_err(|error| {
			ToolError::Fatal(format!("web-search provider returned invalid URL: {error}"))
		})?;
	}
	Ok(())
}

#[derive(Debug, Clone, Copy)]
struct DateTimeTool;

/// Construct opt-in local `datetime`.
pub fn datetime_tool() -> Arc<dyn AgentTool> {
	Arc::new(DateTimeTool)
}

#[async_trait]
impl AgentTool for DateTimeTool {
	fn implementation_identity(&self) -> String {
		"emelex.datetime@1".to_string()
	}

	fn definition(&self) -> ToolDefinition {
		ToolDefinition::new(
			"datetime",
			"Return current RFC 3339 time at a fixed UTC offset.",
			serde_json::json!({
				"type": "object",
				"properties": {
					"utc_offset_minutes": {
						"type": "integer",
						"minimum": -840,
						"maximum": 840
					}
				},
				"additionalProperties": false
			}),
		)
	}

	fn approval_requirement(
		&self,
		_context: &ToolContext,
		_arguments: &serde_json::Value,
	) -> ApprovalRequirement {
		ApprovalRequirement::None
	}

	async fn invoke(
		&self,
		_context: &ToolContext,
		arguments: serde_json::Value,
	) -> Result<ToolOutput, ToolError> {
		let args: DateTimeArgs = parse_args(arguments)?;
		if !(-840..=840).contains(&args.utc_offset_minutes) {
			return Err(respond("utc_offset_minutes must be in -840..=840"));
		}
		let seconds = args
			.utc_offset_minutes
			.checked_mul(60)
			.ok_or_else(|| respond("UTC offset overflow"))?;
		let offset = FixedOffset::east_opt(seconds)
			.ok_or_else(|| respond("UTC offset is outside chrono bounds"))?;
		let timestamp = Utc::now()
			.with_timezone(&offset)
			.to_rfc3339_opts(SecondsFormat::Secs, true);
		Ok(ToolOutput::success(timestamp))
	}
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DateTimeArgs {
	#[serde(default)]
	utc_offset_minutes: i32,
}

fn validate_http_url(input: &str) -> Result<Url, ToolError> {
	let url = Url::parse(input).map_err(|error| respond(format!("invalid URL: {error}")))?;
	if !matches!(url.scheme(), "http" | "https") {
		return Err(respond("URL scheme must be http or https"));
	}
	if !url.username().is_empty() || url.password().is_some() {
		return Err(respond("URL must not contain credentials"));
	}
	if url.host_str().is_none() {
		return Err(respond("URL must contain a host"));
	}
	Ok(url)
}

fn parse_args<T: serde::de::DeserializeOwned>(
	arguments: serde_json::Value,
) -> Result<T, ToolError> {
	serde_json::from_value(arguments)
		.map_err(|error| respond(format!("invalid tool arguments: {error}")))
}

fn respond(message: impl Into<String>) -> ToolError {
	ToolError::RespondToModel(message.into())
}

#[cfg(test)]
mod tests {
	#![allow(clippy::expect_used, clippy::unwrap_used)]

	use std::io::{Read as _, Write as _};

	use super::*;

	#[test]
	fn url_validation_rejects_credentials_and_non_http_schemes() {
		assert!(validate_http_url("file:///tmp/a").is_err());
		assert!(validate_http_url("https://user:pass@example.com").is_err());
		assert!(validate_http_url("https://example.com/path").is_ok());
	}

	#[test]
	fn search_result_validation_is_bounded() {
		let valid = vec![WebSearchResult {
			title: "Result".to_string(),
			url: "https://example.com".to_string(),
			snippet: "Excerpt".to_string(),
		}];
		let invalid = vec![WebSearchResult {
			title: "Result".to_string(),
			url: "file:///tmp/a".to_string(),
			snippet: "Excerpt".to_string(),
		}];

		assert!(validate_search_results(&valid).is_ok());
		assert!(validate_search_results(&invalid).is_err());
	}

	#[tokio::test]
	async fn datetime_uses_requested_fixed_offset() {
		let tool = DateTimeTool;
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(
				super::super::workspace::WorkspaceRoot::open(directory.path()).expect("workspace"),
			),
			cancellation: super::super::AgentCancellation::new(),
			approved: false,
		};
		let output = tool
			.invoke(&context, serde_json::json!({"utc_offset_minutes": 120}))
			.await
			.expect("datetime");

		assert!(output.content.ends_with("+02:00"));
	}

	#[tokio::test]
	async fn redirect_requires_new_tool_call_and_approval() {
		let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
		let address = listener.local_addr().expect("address");
		let server = std::thread::spawn(move || {
			let (mut stream, _) = listener.accept().expect("accept");
			let mut request = [0_u8; 1024];
			let _ = stream.read(&mut request).expect("request");
			stream
				.write_all(
					b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
				)
				.expect("response");
		});
		let directory = tempfile::tempdir().expect("tempdir");
		let context = ToolContext {
			call_id: uuid::Uuid::now_v7().to_string(),
			workspace: Arc::new(
				super::super::workspace::WorkspaceRoot::open(directory.path()).expect("workspace"),
			),
			cancellation: super::super::AgentCancellation::new(),
			approved: true,
		};
		let tool = web_fetch_tool().expect("web fetch");

		let output = tool
			.invoke(
				&context,
				serde_json::json!({"url": format!("http://{address}/start")}),
			)
			.await
			.expect("redirect output");
		server.join().expect("server");

		assert!(output.is_error);
		assert!(output.content.contains("/next"));
		assert!(output.content.contains("new web_fetch call"));
	}
}
