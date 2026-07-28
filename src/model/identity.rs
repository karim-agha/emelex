//! Validated model identity types.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

const MAX_HUB_COMPONENT_BYTES: usize = 96;
const MAX_HUB_ID_BYTES: usize = 96;
const MAX_LOCAL_NAME_BYTES: usize = 128;

/// A Hugging Face repository address in `repo_name` or
/// `namespace/repo_name` form.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HubModelId(String);

impl HubModelId {
	/// Parse and validate `repo_name` or `namespace/repo_name`.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] unless the value follows Hugging Face's
	/// one- or two-component repository-ID grammar.
	#[expect(
		clippy::case_sensitive_file_extension_comparisons,
		reason = "case-sensitive suffix matches Hugging Face's repository ID validator"
	)]
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into();
		let (namespace, repo_name) = value
			.split_once('/')
			.map_or((None, value.as_str()), |(namespace, repo_name)| {
				(Some(namespace), repo_name)
			});
		if value.is_empty()
			|| repo_name.is_empty()
			|| namespace.is_some_and(str::is_empty)
			|| repo_name.contains('/')
			|| namespace.is_some_and(|namespace| namespace.contains('/'))
			|| value.len() > MAX_HUB_ID_BYTES
			|| !valid_hub_component(repo_name)
			|| namespace.is_some_and(|namespace| !valid_hub_component(namespace))
			|| value.contains("--")
			|| value.contains("..")
			|| value.ends_with(".git")
		{
			return Err(ModelRefError::InvalidHub(value));
		}
		Ok(Self(value))
	}

	/// Original validated repository ID.
	pub fn as_str(&self) -> &str {
		&self.0
	}

	/// Optional username or organization namespace.
	pub fn namespace(&self) -> Option<&str> {
		self.0.split_once('/').map(|(namespace, _)| namespace)
	}

	/// Repository name without its optional namespace.
	pub fn repo_name(&self) -> &str {
		self.0
			.split_once('/')
			.map_or(self.as_str(), |(_, repo_name)| repo_name)
	}
}

impl fmt::Display for HubModelId {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

impl FromStr for HubModelId {
	type Err = ModelRefError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		Self::parse(value)
	}
}

impl Serialize for HubModelId {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(self.as_str())
	}
}

impl<'de> Deserialize<'de> for HubModelId {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = String::deserialize(deserializer)?;
		Self::parse(value).map_err(serde::de::Error::custom)
	}
}

/// Stable name of a model copied into Emelex from local storage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalModelName(String);

impl LocalModelName {
	/// Parse a local model name without the `local:` prefix.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] for empty, path-like, or unsafe names.
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into();
		if !valid_component(&value, MAX_LOCAL_NAME_BYTES) {
			return Err(ModelRefError::InvalidLocal(value));
		}
		Ok(Self(value))
	}

	/// Validated local name.
	pub fn as_str(&self) -> &str {
		&self.0
	}
}

impl fmt::Display for LocalModelName {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

impl Serialize for LocalModelName {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(self.as_str())
	}
}

impl<'de> Deserialize<'de> for LocalModelName {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = String::deserialize(deserializer)?;
		Self::parse(value).map_err(serde::de::Error::custom)
	}
}

/// Address of either a Hugging Face model or an imported local model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ModelRef {
	/// Hugging Face repository.
	Hub(HubModelId),
	/// Emelex-owned local import.
	Local(LocalModelName),
}

impl ModelRef {
	/// Parse `repo_name`, `namespace/repo_name`, or `local:<name>`.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] when the syntax is invalid.
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into();
		if let Some(name) = value.strip_prefix("local:") {
			return LocalModelName::parse(name.to_string()).map(Self::Local);
		}
		HubModelId::parse(value).map(Self::Hub)
	}

	/// Hub identity when this reference is online-addressable.
	pub const fn as_hub(&self) -> Option<&HubModelId> {
		match self {
			Self::Hub(id) => Some(id),
			Self::Local(_) => None,
		}
	}
}

impl fmt::Display for ModelRef {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Hub(id) => id.fmt(formatter),
			Self::Local(name) => write!(formatter, "local:{name}"),
		}
	}
}

impl FromStr for ModelRef {
	type Err = ModelRefError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		Self::parse(value)
	}
}

impl Serialize for ModelRef {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(&self.to_string())
	}
}

impl<'de> Deserialize<'de> for ModelRef {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = String::deserialize(deserializer)?;
		Self::parse(value).map_err(serde::de::Error::custom)
	}
}

/// Immutable Hugging Face commit identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ResolvedRevision(String);

impl ResolvedRevision {
	/// Validate a 40-character Git commit SHA.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] when `value` is not a full SHA.
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into().to_ascii_lowercase();
		if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
			return Err(ModelRefError::InvalidRevision(value));
		}
		Ok(Self(value))
	}

	/// Full lowercase commit SHA.
	pub fn as_str(&self) -> &str {
		&self.0
	}
}

impl TryFrom<String> for ResolvedRevision {
	type Error = ModelRefError;

	fn try_from(value: String) -> Result<Self, Self::Error> {
		Self::parse(value)
	}
}

impl From<ResolvedRevision> for String {
	fn from(value: ResolvedRevision) -> Self {
		value.0
	}
}

impl fmt::Display for ResolvedRevision {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

/// Content digest identifying one immutable local model snapshot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SnapshotDigest(String);

impl SnapshotDigest {
	/// Validate a lowercase or uppercase SHA-256 digest.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] unless `value` contains exactly 64
	/// hexadecimal characters.
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into().to_ascii_lowercase();
		if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
			return Err(ModelRefError::InvalidSnapshotDigest(value));
		}
		Ok(Self(value))
	}

	/// Full lowercase SHA-256.
	pub fn as_str(&self) -> &str {
		&self.0
	}
}

impl TryFrom<String> for SnapshotDigest {
	type Error = ModelRefError;

	fn try_from(value: String) -> Result<Self, Self::Error> {
		Self::parse(value)
	}
}

impl From<SnapshotDigest> for String {
	fn from(value: SnapshotDigest) -> Self {
		value.0
	}
}

impl fmt::Display for SnapshotDigest {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.as_str())
	}
}

/// Exact immutable installed-model address.
///
/// Stable Hub references can advance to newer snapshots.
/// A snapshot ID always includes the immutable Hub commit or local content
/// digest and is therefore suitable for durable session bindings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ModelSnapshotId {
	/// Hub repository at one exact commit.
	Hub {
		/// Repository identity.
		id: HubModelId,
		/// Full immutable commit.
		revision: ResolvedRevision,
	},
	/// Emelex-owned local import at one exact content digest.
	Local {
		/// Stable local name.
		name: LocalModelName,
		/// Digest over the immutable runtime-file inventory.
		digest: SnapshotDigest,
	},
}

impl ModelSnapshotId {
	/// Parse `<repo-id>@<commit>` or `local:<name>@<sha256>`.
	///
	/// # Errors
	///
	/// Returns [`ModelRefError`] when either the stable reference or exact
	/// snapshot component is malformed.
	pub fn parse(value: impl Into<String>) -> Result<Self, ModelRefError> {
		let value = value.into();
		let (reference, exact) = value
			.rsplit_once('@')
			.ok_or_else(|| ModelRefError::InvalidSnapshot(value.clone()))?;
		let reference = ModelRef::parse(reference.to_string())?;
		match reference {
			ModelRef::Hub(id) => Ok(Self::Hub {
				id,
				revision: ResolvedRevision::parse(exact.to_string())?,
			}),
			ModelRef::Local(name) => Ok(Self::Local {
				name,
				digest: SnapshotDigest::parse(exact.to_string())?,
			}),
		}
	}

	/// Stable reference shared by successive snapshots.
	pub fn reference(&self) -> ModelRef {
		match self {
			Self::Hub { id, .. } => ModelRef::Hub(id.clone()),
			Self::Local { name, .. } => ModelRef::Local(name.clone()),
		}
	}
}

impl fmt::Display for ModelSnapshotId {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Hub { id, revision } => write!(formatter, "{id}@{revision}"),
			Self::Local { name, digest } => write!(formatter, "local:{name}@{digest}"),
		}
	}
}

impl FromStr for ModelSnapshotId {
	type Err = ModelRefError;

	fn from_str(value: &str) -> Result<Self, Self::Err> {
		Self::parse(value)
	}
}

impl Serialize for ModelSnapshotId {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_str(&self.to_string())
	}
}

impl<'de> Deserialize<'de> for ModelSnapshotId {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		let value = String::deserialize(deserializer)?;
		Self::parse(value).map_err(serde::de::Error::custom)
	}
}

/// Model identity validation failures.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[allow(
	clippy::enum_variant_names,
	reason = "public validation errors read clearly with the Invalid prefix"
)]
pub enum ModelRefError {
	/// Hub ID is not `repo_name` or `namespace/repo_name`.
	#[error("invalid Hugging Face model ID {0:?}; expected repo_name or namespace/repo_name")]
	InvalidHub(String),
	/// Local name is unsafe or path-like.
	#[error("invalid local model name {0:?}")]
	InvalidLocal(String),
	/// Revision is not a full commit SHA.
	#[error("invalid resolved revision {0:?}; expected a 40-character commit SHA")]
	InvalidRevision(String),
	/// Local snapshot digest is not a full SHA-256.
	#[error("invalid snapshot digest {0:?}; expected a 64-character SHA-256")]
	InvalidSnapshotDigest(String),
	/// Exact snapshot address lacks a separator or has malformed syntax.
	#[error(
		"invalid model snapshot ID {0:?}; expected <repo-id>@<commit> or local:<name>@<sha256>"
	)]
	InvalidSnapshot(String),
}

fn valid_component(value: &str, max_bytes: usize) -> bool {
	!value.is_empty()
		&& value.len() <= max_bytes
		&& value != "."
		&& value != ".."
		&& !value.starts_with('.')
		&& !value.ends_with('.')
		&& value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn valid_hub_component(value: &str) -> bool {
	value.len() <= MAX_HUB_COMPONENT_BYTES
		&& value
			.as_bytes()
			.first()
			.is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
		&& value
			.as_bytes()
			.last()
			.is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
		&& value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
