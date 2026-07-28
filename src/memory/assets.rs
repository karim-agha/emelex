//! Descriptor-safe content-addressed assets for durable Session media.

use std::{
	ffi::{CStr, CString},
	fs::{File, OpenOptions},
	io::{Read, Write as _},
	mem::MaybeUninit,
	os::{
		fd::{AsRawFd as _, FromRawFd as _},
		unix::{
			ffi::OsStrExt as _,
			fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _},
		},
	},
	path::{Path, PathBuf},
	time::Duration,
};

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension as _, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use super::{MemoryError, MemoryStore, SessionLease, current_user_id};

/// Maximum bytes accepted for one durable asset.
pub const MAX_ASSET_BYTES: u64 = 128 << 20;

const ASSETS_DIRECTORY: &str = "assets";
const MAX_EVENT_ASSETS: usize = 1_024;
const MAX_SESSION_ASSETS: usize = 100_000;
const MAX_GC_CATALOG_ROWS: usize = 256;
const MAX_GC_DIRECTORY_ENTRIES: usize = 4_096;
const IO_BUFFER_BYTES: usize = 64 << 10;
const TEMP_PREFIX: &str = ".asset-";
const ACL_TYPE_EXTENDED: libc::c_int = 0x0000_0100;
const ACL_FIRST_ENTRY: libc::c_int = 0;

type Acl = *mut libc::c_void;
type AclEntry = *mut libc::c_void;

unsafe extern "C" {
	fn acl_free(object: *mut libc::c_void) -> libc::c_int;
	fn acl_delete_entry(acl: Acl, entry: AclEntry) -> libc::c_int;
	fn acl_get_fd_np(descriptor: libc::c_int, acl_type: libc::c_int) -> Acl;
	fn acl_get_entry(acl: Acl, entry_id: libc::c_int, entry: *mut AclEntry) -> libc::c_int;
	fn acl_set_fd_np(descriptor: libc::c_int, acl: Acl, acl_type: libc::c_int) -> libc::c_int;
}

struct OwnedAcl(Acl);

impl Drop for OwnedAcl {
	fn drop(&mut self) {
		// SAFETY: this guard owns the ACL allocated by an acl_* function.
		unsafe {
			acl_free(self.0);
		}
	}
}

/// Media category carried by one event-local asset reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AssetKind {
	/// Encoded image bytes.
	Image,
	/// Encoded audio bytes.
	Audio,
	/// Encoded video bytes.
	Video,
	/// Non-media or embedding-defined bytes.
	Other,
}

impl AssetKind {
	/// Stable lowercase storage spelling.
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Image => "image",
			Self::Audio => "audio",
			Self::Video => "video",
			Self::Other => "other",
		}
	}

	pub(super) fn parse(value: &str) -> Result<Self, MemoryError> {
		match value {
			"image" => Ok(Self::Image),
			"audio" => Ok(Self::Audio),
			"video" => Ok(Self::Video),
			"other" => Ok(Self::Other),
			_ => Err(MemoryError::Corrupt(format!(
				"unknown durable asset kind {value:?}"
			))),
		}
	}
}

/// Typed reference to immutable content-addressed bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[non_exhaustive]
pub struct AssetRef {
	sha256: String,
	bytes: u64,
	kind: AssetKind,
}

impl AssetRef {
	pub(super) fn new(sha256: String, bytes: u64, kind: AssetKind) -> Result<Self, MemoryError> {
		validate_sha256(&sha256)?;
		if bytes > MAX_ASSET_BYTES {
			return Err(MemoryError::Invalid(format!(
				"asset byte count exceeds {MAX_ASSET_BYTES} byte limit"
			)));
		}
		Ok(Self {
			sha256,
			bytes,
			kind,
		})
	}

	/// Lowercase SHA-256 content identity.
	pub fn sha256(&self) -> &str {
		&self.sha256
	}

	/// Exact byte count.
	pub const fn bytes(&self) -> u64 {
		self.bytes
	}

	/// Event-local media category.
	pub const fn kind(&self) -> AssetKind {
		self.kind
	}
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAssetRef {
	sha256: String,
	bytes: u64,
	kind: AssetKind,
}

impl<'de> Deserialize<'de> for AssetRef {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let raw = RawAssetRef::deserialize(deserializer)?;
		Self::new(raw.sha256, raw.bytes, raw.kind).map_err(D::Error::custom)
	}
}

/// Catalog metadata for one verified asset reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssetRecord {
	/// Content reference.
	pub reference: AssetRef,
	/// First catalog insertion time.
	pub created_at: DateTime<Utc>,
	/// Last successful full-file digest verification.
	pub verified_at: DateTime<Utc>,
}

/// Result of one bounded asset collection pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssetGcReport {
	/// Cataloged, unreferenced files removed.
	pub cataloged_files: usize,
	/// Files without catalog rows removed.
	pub orphan_files: usize,
	/// Total bytes removed when known.
	pub bytes_removed: u64,
}

impl MemoryStore {
	/// Directory containing content-addressed asset files.
	pub fn assets_dir(&self) -> PathBuf {
		assets_path(&self.database)
	}

	/// Store one bounded stream and return its typed content reference.
	///
	/// The stream is hashed while writing an owner-only temporary file. The
	/// file is synced and published without clobbering before its catalog row
	/// commits.
	///
	/// # Errors
	///
	/// Returns input, filesystem, corruption, or database errors.
	pub fn store_asset(
		&self,
		kind: AssetKind,
		mut source: impl std::io::Read,
	) -> Result<AssetRef, MemoryError> {
		let directory_path = self.assets_dir();
		let directory = open_assets_directory(&self.database)?;
		let mut temporary = PendingAsset::create(&directory, &directory_path)?;
		let temporary_path = temporary.display().to_path_buf();
		let (sha256, bytes) =
			write_bounded_asset(&mut source, temporary.file_mut(), &temporary_path)?;
		temporary
			.file()
			.sync_all()
			.map_err(|source| io_error("sync temporary asset", temporary.display(), source))?;

		let reference = AssetRef::new(sha256, bytes, kind)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		publish_or_verify(&directory, &directory_path, &mut temporary, &reference)?;
		directory.sync_all().map_err(|source| {
			io_error(
				"sync asset directory after publication",
				&directory_path,
				source,
			)
		})?;
		let now = Utc::now().to_rfc3339();
		let changed = transaction.execute(
			"INSERT INTO assets (sha256, bytes, created_at, verified_at)
			 VALUES (?1, ?2, ?3, ?3)
			 ON CONFLICT(sha256) DO UPDATE SET verified_at = excluded.verified_at
			 WHERE assets.bytes = excluded.bytes",
			params![reference.sha256(), sql_bytes(reference.bytes())?, &now,],
		)?;
		if changed != 1 {
			return Err(MemoryError::Corrupt(format!(
				"asset {} catalog byte count conflicts with verified content",
				reference.sha256()
			)));
		}
		transaction.commit()?;
		Ok(reference)
	}

	/// Store one in-memory byte slice.
	///
	/// # Errors
	///
	/// Returns input, filesystem, corruption, or database errors.
	pub fn store_asset_bytes(
		&self,
		kind: AssetKind,
		bytes: &[u8],
	) -> Result<AssetRef, MemoryError> {
		self.store_asset(kind, bytes)
	}

	/// Read and fully verify one cataloged asset.
	///
	/// # Errors
	///
	/// Returns an error when the reference is invalid, the catalog entry is
	/// absent, file metadata is unsafe, bytes changed, or storage fails.
	pub fn read_asset(&self, reference: &AssetRef) -> Result<Vec<u8>, MemoryError> {
		validate_reference(reference)?;
		let directory_path = self.assets_dir();
		let directory = open_assets_directory(&self.database)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let bytes = read_verified_asset(&transaction, &directory, &directory_path, reference)?;
		transaction.commit()?;
		Ok(bytes)
	}

	pub(super) fn read_event_asset(
		&self,
		event_id: Uuid,
		ordinal: usize,
		reference: &AssetRef,
	) -> Result<Vec<u8>, MemoryError> {
		validate_reference(reference)?;
		let ordinal = i64::try_from(ordinal)
			.map_err(|_| MemoryError::Corrupt("event asset ordinal is too large".to_string()))?;
		let directory_path = self.assets_dir();
		let directory = open_assets_directory(&self.database)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let linked: bool = transaction.query_row(
			"SELECT EXISTS(
			   SELECT 1 FROM session_assets
			   WHERE event_id = ?1 AND ordinal = ?2
			     AND asset_sha256 = ?3 AND kind = ?4
			 )",
			params![
				event_id.to_string(),
				ordinal,
				reference.sha256(),
				reference.kind().as_str(),
			],
			|row| row.get(0),
		)?;
		if !linked {
			return Err(MemoryError::Corrupt(format!(
				"event {event_id} asset {ordinal} does not match its durable linkage"
			)));
		}
		let bytes = read_verified_asset(&transaction, &directory, &directory_path, reference)?;
		transaction.commit()?;
		Ok(bytes)
	}

	pub(super) fn verify_event_asset_count(
		&self,
		event_id: Uuid,
		expected: usize,
	) -> Result<(), MemoryError> {
		let expected = u64::try_from(expected)
			.map_err(|_| MemoryError::Corrupt("event asset count is too large".to_string()))?;
		let actual: i64 = self.connection()?.query_row(
			"SELECT COUNT(*) FROM session_assets WHERE event_id = ?1",
			[event_id.to_string()],
			|row| row.get(0),
		)?;
		let actual = u64::try_from(actual)
			.map_err(|_| MemoryError::Corrupt("event has a negative asset count".to_string()))?;
		if actual != expected {
			return Err(MemoryError::Corrupt(format!(
				"event {event_id} carries {expected} asset references but has {actual} durable links"
			)));
		}
		Ok(())
	}

	/// List a Session's event-linked assets in transcript and payload order.
	///
	/// # Errors
	///
	/// Returns not-found, corruption, or database errors.
	pub fn session_assets(&self, session_id: Uuid) -> Result<Vec<AssetRef>, MemoryError> {
		let connection = self.connection()?;
		let exists: bool = connection.query_row(
			"SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
			[session_id.to_string()],
			|row| row.get(0),
		)?;
		if !exists {
			return Err(MemoryError::NotFound {
				entity: "session",
				id: session_id,
			});
		}
		let mut statement = connection.prepare(
			"SELECT sa.asset_sha256, a.bytes, sa.kind
			 FROM session_assets sa
			 JOIN session_events e ON e.id = sa.event_id
			 JOIN assets a ON a.sha256 = sa.asset_sha256
			 WHERE sa.session_id = ?1
			 ORDER BY e.sequence ASC, sa.ordinal ASC
			 LIMIT ?2",
		)?;
		let limit = i64::try_from(MAX_SESSION_ASSETS + 1)
			.map_err(|_| MemoryError::Invalid("Session asset limit is invalid".to_string()))?;
		let rows = statement.query_map(params![session_id.to_string(), limit], |row| {
			Ok((
				row.get::<_, String>(0)?,
				row.get::<_, i64>(1)?,
				row.get::<_, String>(2)?,
			))
		})?;
		let mut assets = Vec::new();
		for row in rows {
			if assets.len() == MAX_SESSION_ASSETS {
				return Err(MemoryError::Invalid(format!(
					"Session assets exceed {MAX_SESSION_ASSETS} item limit"
				)));
			}
			let (sha256, bytes, kind) = row?;
			assets.push(AssetRef::new(
				sha256,
				stored_bytes(bytes)?,
				AssetKind::parse(&kind)?,
			)?);
		}
		Ok(assets)
	}

	/// Collect bounded unreferenced and orphaned assets older than `grace`.
	///
	/// Publication and deletion coordinate through `SQLite` write locks. A crash
	/// can therefore leave an orphan for a later pass, but never a committed
	/// catalog row whose successfully published file was collected.
	///
	/// # Errors
	///
	/// Returns invalid-retention, filesystem, corruption, or database errors.
	pub fn gc_assets(&self, grace: Duration) -> Result<AssetGcReport, MemoryError> {
		if grace.is_zero() {
			return Err(MemoryError::Invalid(
				"asset collection grace must be positive".to_string(),
			));
		}
		let grace = chrono::Duration::from_std(grace)
			.map_err(|_| MemoryError::Invalid("asset collection grace is invalid".to_string()))?;
		let cutoff = Utc::now()
			.checked_sub_signed(grace)
			.ok_or_else(|| MemoryError::Invalid("asset collection cutoff overflow".to_string()))?;
		let directory_path = self.assets_dir();
		let directory = open_assets_directory(&self.database)?;
		let candidates = self.gc_catalog_candidates(cutoff)?;
		let mut report = AssetGcReport::default();
		for candidate in candidates {
			if let Some(bytes) =
				self.remove_cataloged_asset(&directory, &directory_path, &candidate, cutoff)?
			{
				report.cataloged_files += 1;
				report.bytes_removed = checked_removed_bytes(report.bytes_removed, bytes)?;
			}
		}
		self.remove_orphan_assets(&directory, &directory_path, cutoff, &mut report)?;
		Ok(report)
	}

	fn gc_catalog_candidates(
		&self,
		cutoff: DateTime<Utc>,
	) -> Result<Vec<CatalogCandidate>, MemoryError> {
		let connection = self.connection()?;
		let mut statement = connection.prepare(
			"SELECT a.sha256, a.bytes, a.created_at, a.verified_at
			 FROM assets a
			 WHERE a.created_at <= ?1
				   AND NOT EXISTS (
				     SELECT 1 FROM session_assets sa
				     WHERE sa.asset_sha256 = a.sha256
				   )
				   AND NOT EXISTS (
				     SELECT 1 FROM pending_tool_assets pa
				     WHERE pa.asset_sha256 = a.sha256
				   )
				   AND NOT EXISTS (
				     SELECT 1 FROM active_agent_turn_assets aa
				     WHERE aa.asset_sha256 = a.sha256
				   )
			 ORDER BY a.created_at ASC, a.sha256 ASC
			 LIMIT ?2",
		)?;
		let rows = statement.query_map(
			params![
				cutoff.to_rfc3339(),
				i64::try_from(MAX_GC_CATALOG_ROWS).map_err(|_| {
					MemoryError::Invalid("asset GC row limit is invalid".to_string())
				})?,
			],
			|row| {
				Ok(CatalogCandidate {
					sha256: row.get(0)?,
					bytes: row.get(1)?,
					created_at: row.get(2)?,
					verified_at: row.get(3)?,
				})
			},
		)?;
		rows.collect::<Result<Vec<_>, _>>()
			.map_err(MemoryError::from)
	}

	fn remove_cataloged_asset(
		&self,
		directory: &File,
		directory_path: &Path,
		candidate: &CatalogCandidate,
		cutoff: DateTime<Utc>,
	) -> Result<Option<u64>, MemoryError> {
		validate_sha256(&candidate.sha256)?;
		let bytes = stored_bytes(candidate.bytes)?;
		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let removed = transaction.execute(
			"DELETE FROM assets
			 WHERE sha256 = ?1 AND bytes = ?2 AND created_at = ?3
			   AND created_at <= ?4
				   AND NOT EXISTS (
				     SELECT 1 FROM session_assets sa
				     WHERE sa.asset_sha256 = assets.sha256
				   )
				   AND NOT EXISTS (
				     SELECT 1 FROM pending_tool_assets pa
				     WHERE pa.asset_sha256 = assets.sha256
				   )
				   AND NOT EXISTS (
				     SELECT 1 FROM active_agent_turn_assets aa
				     WHERE aa.asset_sha256 = assets.sha256
				   )",
			params![
				&candidate.sha256,
				candidate.bytes,
				&candidate.created_at,
				cutoff.to_rfc3339(),
			],
		)?;
		transaction.commit()?;
		if removed == 0 {
			return Ok(None);
		}

		let mut connection = self.connection()?;
		let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
		let recataloged: bool = transaction.query_row(
			"SELECT EXISTS(SELECT 1 FROM assets WHERE sha256 = ?1)",
			[&candidate.sha256],
			|row| row.get(0),
		)?;
		if recataloged {
			transaction.commit()?;
			return Ok(None);
		}
		let mut file = match open_asset_file(directory, directory_path, &candidate.sha256) {
			Ok(Some(file)) => file,
			Ok(None) => {
				transaction.commit()?;
				return Ok(None);
			}
			Err(error) => {
				restore_catalog_row(&transaction, candidate)?;
				transaction.commit()?;
				return Err(error);
			}
		};
		let path = asset_path(directory_path, &candidate.sha256);
		if let Err(error) = validate_asset_metadata(&file, &path, Some(bytes))
			.and_then(|_| verify_digest(&mut file, &path, &candidate.sha256, bytes))
		{
			restore_catalog_row(&transaction, candidate)?;
			transaction.commit()?;
			return Err(error);
		}
		if let Err(error) = unlink_open_file(directory, &candidate.sha256, &file, &path) {
			restore_catalog_row(&transaction, candidate)?;
			transaction.commit()?;
			return Err(error);
		}
		directory.sync_all().map_err(|source| {
			io_error(
				"sync asset directory after collection",
				directory_path,
				source,
			)
		})?;
		transaction.commit()?;
		Ok(Some(bytes))
	}

	fn remove_orphan_assets(
		&self,
		directory: &File,
		directory_path: &Path,
		cutoff: DateTime<Utc>,
		report: &mut AssetGcReport,
	) -> Result<(), MemoryError> {
		let mut stream = DirectoryStream::open(directory, directory_path)?;
		let temporary_cutoff = Utc::now() - chrono::Duration::hours(1);
		for _ in 0..MAX_GC_DIRECTORY_ENTRIES {
			let Some(name) = stream.next(directory_path)? else {
				break;
			};
			let Ok(name) = std::str::from_utf8(&name) else {
				continue;
			};
			let digest = is_sha256(name);
			if !digest && !is_canonical_temp(name) {
				continue;
			}
			let mut connection = self.connection()?;
			let transaction =
				connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
			if digest {
				let cataloged: bool = transaction.query_row(
					"SELECT EXISTS(SELECT 1 FROM assets WHERE sha256 = ?1)",
					[name],
					|row| row.get(0),
				)?;
				if cataloged {
					transaction.commit()?;
					continue;
				}
			}
			let Some(mut file) = open_asset_file(directory, directory_path, name)? else {
				transaction.commit()?;
				continue;
			};
			let path = asset_path(directory_path, name);
			let metadata = validate_asset_metadata(&file, &path, None)?;
			let file_cutoff = if digest {
				cutoff
			} else {
				cutoff.min(temporary_cutoff)
			};
			if metadata.mtime() > file_cutoff.timestamp() {
				transaction.commit()?;
				continue;
			}
			if !digest && !try_lock_exclusive(&file, &path)? {
				transaction.commit()?;
				continue;
			}
			let bytes = metadata.len();
			if digest {
				verify_digest(&mut file, &path, name, bytes)?;
			}
			unlink_open_file(directory, name, &file, &path)?;
			directory.sync_all().map_err(|source| {
				io_error(
					"sync asset directory after orphan collection",
					directory_path,
					source,
				)
			})?;
			transaction.commit()?;
			report.orphan_files += 1;
			report.bytes_removed = checked_removed_bytes(report.bytes_removed, bytes)?;
		}
		Ok(())
	}
}

pub(super) fn record_event_assets(
	transaction: &Transaction<'_>,
	lease: &SessionLease,
	event_id: Uuid,
	assets: &[AssetRef],
	now: DateTime<Utc>,
) -> Result<(), MemoryError> {
	if assets.len() > MAX_EVENT_ASSETS {
		return Err(MemoryError::Invalid(format!(
			"event assets exceed {MAX_EVENT_ASSETS} item limit"
		)));
	}
	for (ordinal, reference) in assets.iter().enumerate() {
		validate_reference(reference)?;
		let catalog_bytes = transaction
			.query_row(
				"SELECT bytes FROM assets WHERE sha256 = ?1",
				[reference.sha256()],
				|row| row.get::<_, i64>(0),
			)
			.optional()?
			.ok_or_else(|| {
				MemoryError::Invalid(format!(
					"event references uncataloged asset {}",
					reference.sha256()
				))
			})?;
		if stored_bytes(catalog_bytes)? != reference.bytes() {
			return Err(MemoryError::Corrupt(format!(
				"event asset {} byte count conflicts with the catalog",
				reference.sha256()
			)));
		}
		transaction.execute(
			"INSERT INTO session_assets
			 (session_id, event_id, asset_sha256, kind, ordinal, created_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
			params![
				lease.session.id.to_string(),
				event_id.to_string(),
				reference.sha256(),
				reference.kind().as_str(),
				i64::try_from(ordinal)
					.map_err(|_| MemoryError::Invalid("asset ordinal is too large".to_string()))?,
				now.to_rfc3339(),
			],
		)?;
	}
	Ok(())
}

fn read_verified_asset(
	transaction: &Transaction<'_>,
	directory: &File,
	directory_path: &Path,
	reference: &AssetRef,
) -> Result<Vec<u8>, MemoryError> {
	let catalog_bytes = transaction
		.query_row(
			"SELECT bytes FROM assets WHERE sha256 = ?1",
			[reference.sha256()],
			|row| row.get::<_, i64>(0),
		)
		.optional()?
		.ok_or_else(|| {
			MemoryError::Invalid(format!(
				"asset {} is not present in the durable catalog",
				reference.sha256()
			))
		})?;
	let catalog_bytes = stored_bytes(catalog_bytes)?;
	if catalog_bytes != reference.bytes() {
		return Err(MemoryError::Corrupt(format!(
			"asset {} reference byte count does not match its catalog row",
			reference.sha256()
		)));
	}
	let path = asset_path(directory_path, reference.sha256());
	let mut file =
		open_asset_file(directory, directory_path, reference.sha256())?.ok_or_else(|| {
			MemoryError::Corrupt(format!("cataloged asset {} is missing", reference.sha256()))
		})?;
	validate_asset_metadata(&file, &path, Some(reference.bytes()))?;
	let bytes = read_and_verify(&mut file, &path, reference.sha256(), reference.bytes())?;
	let changed = transaction.execute(
		"UPDATE assets SET verified_at = ?2
		 WHERE sha256 = ?1 AND bytes = ?3",
		params![
			reference.sha256(),
			Utc::now().to_rfc3339(),
			sql_bytes(reference.bytes())?,
		],
	)?;
	if changed != 1 {
		return Err(MemoryError::Corrupt(format!(
			"asset {} catalog changed during verification",
			reference.sha256()
		)));
	}
	Ok(bytes)
}

pub(super) fn prepare_assets_dir(database: &Path) -> Result<(), MemoryError> {
	let parent_path = database.parent().ok_or_else(|| {
		MemoryError::Invalid("memory database has no parent directory".to_string())
	})?;
	let parent = open_directory(parent_path, "open memory directory for asset preparation")?;
	let directory_path = assets_path(database);
	let directory_name = directory_path.file_name().ok_or_else(|| {
		MemoryError::Invalid("asset directory path has no final component".to_string())
	})?;
	let name = c_os_name(directory_name)?;
	// SAFETY: `parent` is an open directory and `name` is one C component.
	let created = if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } == 0 {
		true
	} else {
		let source = std::io::Error::last_os_error();
		if source.kind() != std::io::ErrorKind::AlreadyExists {
			return Err(io_error("create asset directory", &directory_path, source));
		}
		false
	};
	let directory = open_directory_at(&parent, &name).map_err(|source| {
		io_error(
			"open asset directory without following links",
			&directory_path,
			source,
		)
	})?;
	if created {
		clear_extended_acl(&directory).map_err(|source| {
			io_error(
				"clear inherited asset directory ACL",
				&directory_path,
				source,
			)
		})?;
		// SAFETY: this process just created and owns the directory descriptor.
		if unsafe { libc::fchmod(directory.as_raw_fd(), 0o700) } != 0 {
			return Err(io_error(
				"secure new asset directory",
				&directory_path,
				std::io::Error::last_os_error(),
			));
		}
	}
	validate_directory_metadata(&directory, &directory_path)?;
	if created {
		directory
			.sync_all()
			.map_err(|source| io_error("sync new asset directory", &directory_path, source))?;
		parent.sync_all().map_err(|source| {
			io_error(
				"sync memory directory after asset directory creation",
				parent_path,
				source,
			)
		})?;
	}
	Ok(())
}

struct CatalogCandidate {
	sha256: String,
	bytes: i64,
	created_at: String,
	verified_at: String,
}

fn restore_catalog_row(
	transaction: &Transaction<'_>,
	candidate: &CatalogCandidate,
) -> Result<(), MemoryError> {
	transaction.execute(
		"INSERT OR IGNORE INTO assets (sha256, bytes, created_at, verified_at)
		 VALUES (?1, ?2, ?3, ?4)",
		params![
			&candidate.sha256,
			candidate.bytes,
			&candidate.created_at,
			&candidate.verified_at,
		],
	)?;
	Ok(())
}

fn publish_or_verify(
	directory: &File,
	directory_path: &Path,
	temporary: &mut PendingAsset,
	reference: &AssetRef,
) -> Result<(), MemoryError> {
	let target = c_name(reference.sha256())?;
	let temporary_name = temporary.name().ok_or_else(|| {
		MemoryError::Corrupt("temporary asset lost its directory entry".to_string())
	})?;
	// `linkat` publishes a second name for the already-synced inode and fails
	// without clobbering when identical content was published concurrently.
	// SAFETY: both descriptors and one-component names remain live.
	let status = unsafe {
		libc::linkat(
			directory.as_raw_fd(),
			temporary_name.as_ptr(),
			directory.as_raw_fd(),
			target.as_ptr(),
			0,
		)
	};
	if status != 0 {
		let source = std::io::Error::last_os_error();
		if source.kind() != std::io::ErrorKind::AlreadyExists {
			return Err(io_error(
				"publish content-addressed asset",
				asset_path(directory_path, reference.sha256()),
				source,
			));
		}
	}
	let mut published = open_asset_file(directory, directory_path, reference.sha256())?
		.ok_or_else(|| {
			MemoryError::Corrupt(format!(
				"published asset {} disappeared",
				reference.sha256()
			))
		})?;
	let path = asset_path(directory_path, reference.sha256());
	validate_asset_metadata(&published, &path, Some(reference.bytes()))?;
	verify_digest(&mut published, &path, reference.sha256(), reference.bytes())?;
	temporary.unlink()?;
	Ok(())
}

fn write_bounded_asset(
	source: &mut impl Read,
	target: &mut File,
	path: &Path,
) -> Result<(String, u64), MemoryError> {
	let mut hasher = Sha256::new();
	let mut bytes = 0_u64;
	let mut buffer = vec![0_u8; IO_BUFFER_BYTES].into_boxed_slice();
	loop {
		let read = source
			.read(&mut buffer)
			.map_err(|source| io_error("read asset input", path, source))?;
		if read == 0 {
			break;
		}
		let read_u64 = u64::try_from(read)
			.map_err(|_| MemoryError::Invalid("asset read size is invalid".to_string()))?;
		bytes = bytes
			.checked_add(read_u64)
			.ok_or_else(|| MemoryError::Invalid("asset byte count overflow".to_string()))?;
		if bytes > MAX_ASSET_BYTES {
			return Err(MemoryError::Invalid(format!(
				"asset exceeds {MAX_ASSET_BYTES} byte limit"
			)));
		}
		target
			.write_all(&buffer[..read])
			.map_err(|source| io_error("write temporary asset", path, source))?;
		hasher.update(&buffer[..read]);
	}
	Ok((hex::encode(hasher.finalize()), bytes))
}

fn read_and_verify(
	file: &mut File,
	path: &Path,
	expected_sha256: &str,
	expected_bytes: u64,
) -> Result<Vec<u8>, MemoryError> {
	let capacity = usize::try_from(expected_bytes)
		.map_err(|_| MemoryError::Corrupt("asset byte count exceeds platform range".to_string()))?;
	let mut output = Vec::with_capacity(capacity);
	let mut hasher = Sha256::new();
	let mut total = 0_u64;
	let mut buffer = vec![0_u8; IO_BUFFER_BYTES].into_boxed_slice();
	loop {
		let read = file
			.read(&mut buffer)
			.map_err(|source| io_error("read asset", path, source))?;
		if read == 0 {
			break;
		}
		total = total
			.checked_add(
				u64::try_from(read)
					.map_err(|_| MemoryError::Corrupt("asset read size is invalid".to_string()))?,
			)
			.ok_or_else(|| MemoryError::Corrupt("asset byte count overflow".to_string()))?;
		if total > MAX_ASSET_BYTES || total > expected_bytes {
			return Err(MemoryError::Corrupt(format!(
				"asset {expected_sha256} grew beyond its verified byte count"
			)));
		}
		hasher.update(&buffer[..read]);
		output.extend_from_slice(&buffer[..read]);
	}
	verify_totals(path, expected_sha256, expected_bytes, total, hasher)?;
	Ok(output)
}

fn verify_digest(
	file: &mut File,
	path: &Path,
	expected_sha256: &str,
	expected_bytes: u64,
) -> Result<(), MemoryError> {
	let mut hasher = Sha256::new();
	let mut total = 0_u64;
	let mut buffer = vec![0_u8; IO_BUFFER_BYTES].into_boxed_slice();
	loop {
		let read = file
			.read(&mut buffer)
			.map_err(|source| io_error("verify asset", path, source))?;
		if read == 0 {
			break;
		}
		total = total
			.checked_add(
				u64::try_from(read)
					.map_err(|_| MemoryError::Corrupt("asset read size is invalid".to_string()))?,
			)
			.ok_or_else(|| MemoryError::Corrupt("asset byte count overflow".to_string()))?;
		if total > MAX_ASSET_BYTES || total > expected_bytes {
			return Err(MemoryError::Corrupt(format!(
				"asset {expected_sha256} exceeds its verified byte count"
			)));
		}
		hasher.update(&buffer[..read]);
	}
	verify_totals(path, expected_sha256, expected_bytes, total, hasher)
}

fn verify_totals(
	path: &Path,
	expected_sha256: &str,
	expected_bytes: u64,
	actual_bytes: u64,
	hasher: Sha256,
) -> Result<(), MemoryError> {
	let actual_sha256 = hex::encode(hasher.finalize());
	if actual_bytes != expected_bytes || actual_sha256 != expected_sha256 {
		return Err(MemoryError::Corrupt(format!(
			"asset {} failed byte-count or SHA-256 verification at {}",
			expected_sha256,
			path.display()
		)));
	}
	Ok(())
}

fn validate_reference(reference: &AssetRef) -> Result<(), MemoryError> {
	validate_sha256(reference.sha256())?;
	if reference.bytes() > MAX_ASSET_BYTES {
		return Err(MemoryError::Invalid(format!(
			"asset byte count exceeds {MAX_ASSET_BYTES} byte limit"
		)));
	}
	Ok(())
}

fn validate_sha256(value: &str) -> Result<(), MemoryError> {
	if !is_sha256(value) {
		return Err(MemoryError::Invalid(
			"asset SHA-256 must be exactly 64 lowercase hexadecimal characters".to_string(),
		));
	}
	Ok(())
}

fn is_sha256(value: &str) -> bool {
	value.len() == 64
		&& value
			.as_bytes()
			.iter()
			.all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn is_canonical_temp(value: &str) -> bool {
	let Some(uuid) = value.strip_prefix(TEMP_PREFIX) else {
		return false;
	};
	Uuid::parse_str(uuid).is_ok_and(|parsed| parsed.hyphenated().to_string() == uuid)
}

fn sql_bytes(bytes: u64) -> Result<i64, MemoryError> {
	i64::try_from(bytes)
		.map_err(|_| MemoryError::Invalid("asset byte count exceeds SQLite range".to_string()))
}

fn stored_bytes(bytes: i64) -> Result<u64, MemoryError> {
	let bytes = u64::try_from(bytes)
		.map_err(|_| MemoryError::Corrupt("asset catalog has a negative byte count".to_string()))?;
	if bytes > MAX_ASSET_BYTES {
		return Err(MemoryError::Corrupt(format!(
			"asset catalog byte count exceeds {MAX_ASSET_BYTES}"
		)));
	}
	Ok(bytes)
}

fn checked_removed_bytes(total: u64, removed: u64) -> Result<u64, MemoryError> {
	total
		.checked_add(removed)
		.ok_or_else(|| MemoryError::Corrupt("asset GC byte count overflow".to_string()))
}

fn assets_path(database: &Path) -> PathBuf {
	if database.file_name() == Some(std::ffi::OsStr::new("emelex.sqlite3")) {
		return database.with_file_name(ASSETS_DIRECTORY);
	}
	let Some(database_name) = database.file_name() else {
		return database.with_file_name(ASSETS_DIRECTORY);
	};
	let mut directory_name = database_name.to_os_string();
	directory_name.push(".assets");
	database.with_file_name(directory_name)
}

fn asset_path(directory: &Path, name: &str) -> PathBuf {
	directory.join(name)
}

fn open_assets_directory(database: &Path) -> Result<File, MemoryError> {
	let path = assets_path(database);
	let directory = open_directory(&path, "open asset directory")?;
	validate_directory_metadata(&directory, &path)?;
	Ok(directory)
}

fn open_directory(path: &Path, operation: &'static str) -> Result<File, MemoryError> {
	OpenOptions::new()
		.read(true)
		.custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
		.open(path)
		.map_err(|source| io_error(operation, path, source))
}

fn open_directory_at(parent: &File, name: &CStr) -> std::io::Result<File> {
	// SAFETY: `parent` is an open directory and `name` is one C component.
	let descriptor = unsafe {
		libc::openat(
			parent.as_raw_fd(),
			name.as_ptr(),
			libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
		)
	};
	if descriptor < 0 {
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: successful `openat` returned one newly-owned descriptor.
	Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn validate_directory_metadata(directory: &File, path: &Path) -> Result<(), MemoryError> {
	let metadata = directory
		.metadata()
		.map_err(|source| io_error("inspect asset directory", path, source))?;
	if !metadata.is_dir()
		|| metadata.uid() != current_user_id()
		|| metadata.permissions().mode() & 0o7777 != 0o700
		|| has_extended_acl(directory)
			.map_err(|source| io_error("inspect asset directory ACL", path, source))?
	{
		return Err(MemoryError::Invalid(format!(
			"asset directory {} must be current-user-owned, mode 0700, and have no extended ACL",
			path.display()
		)));
	}
	Ok(())
}

fn open_asset_file(
	directory: &File,
	directory_path: &Path,
	name: &str,
) -> Result<Option<File>, MemoryError> {
	let name_c = c_name(name)?;
	// SAFETY: directory is live and name is one NUL-terminated component.
	let descriptor = unsafe {
		libc::openat(
			directory.as_raw_fd(),
			name_c.as_ptr(),
			libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
		)
	};
	if descriptor < 0 {
		let source = std::io::Error::last_os_error();
		if source.kind() == std::io::ErrorKind::NotFound {
			return Ok(None);
		}
		return Err(io_error(
			"open asset without following links",
			asset_path(directory_path, name),
			source,
		));
	}
	// SAFETY: successful `openat` returned one newly-owned descriptor.
	Ok(Some(unsafe { File::from_raw_fd(descriptor) }))
}

fn validate_asset_metadata(
	file: &File,
	path: &Path,
	expected_bytes: Option<u64>,
) -> Result<std::fs::Metadata, MemoryError> {
	let metadata = file
		.metadata()
		.map_err(|source| io_error("inspect asset", path, source))?;
	if !metadata.is_file()
		|| metadata.uid() != current_user_id()
		|| metadata.permissions().mode() & 0o7777 != 0o600
		|| has_extended_acl(file).map_err(|source| io_error("inspect asset ACL", path, source))?
	{
		return Err(MemoryError::Corrupt(format!(
			"asset {} must be current-user-owned, mode 0600, and have no extended ACL",
			path.display()
		)));
	}
	if metadata.len() > MAX_ASSET_BYTES {
		return Err(MemoryError::Corrupt(format!(
			"asset {} exceeds {MAX_ASSET_BYTES} byte limit",
			path.display()
		)));
	}
	if expected_bytes.is_some_and(|expected| metadata.len() != expected) {
		return Err(MemoryError::Corrupt(format!(
			"asset {} metadata byte count differs from its catalog",
			path.display()
		)));
	}
	Ok(metadata)
}

fn unlink_open_file(
	directory: &File,
	name: &str,
	opened: &File,
	path: &Path,
) -> Result<(), MemoryError> {
	let name_c = c_name(name)?;
	let expected = opened
		.metadata()
		.map_err(|source| io_error("inspect opened asset before deletion", path, source))?;
	let actual = metadata_at(directory, &name_c)
		.map_err(|source| io_error("inspect asset name before deletion", path, source))?;
	let actual_device = u64::try_from(actual.st_dev)
		.map_err(|_| MemoryError::Corrupt("asset has a negative device identity".to_string()))?;
	if expected.dev() != actual_device || expected.ino() != actual.st_ino {
		return Err(MemoryError::Corrupt(format!(
			"asset {} changed before deletion",
			path.display()
		)));
	}
	// SAFETY: directory and single-component name are live.
	if unsafe { libc::unlinkat(directory.as_raw_fd(), name_c.as_ptr(), 0) } != 0 {
		return Err(io_error(
			"delete asset",
			path,
			std::io::Error::last_os_error(),
		));
	}
	Ok(())
}

fn metadata_at(directory: &File, name: &CStr) -> std::io::Result<libc::stat> {
	let mut metadata = MaybeUninit::<libc::stat>::uninit();
	// SAFETY: directory and name are live; successful `fstatat` initializes
	// the complete output structure.
	if unsafe {
		libc::fstatat(
			directory.as_raw_fd(),
			name.as_ptr(),
			metadata.as_mut_ptr(),
			libc::AT_SYMLINK_NOFOLLOW,
		)
	} != 0
	{
		return Err(std::io::Error::last_os_error());
	}
	// SAFETY: successful `fstatat` initialized the value.
	Ok(unsafe { metadata.assume_init() })
}

fn c_name(value: &str) -> Result<CString, MemoryError> {
	CString::new(value.as_bytes())
		.map_err(|_| MemoryError::Invalid("asset filename contains an interior NUL".to_string()))
}

fn c_os_name(value: &std::ffi::OsStr) -> Result<CString, MemoryError> {
	CString::new(value.as_bytes())
		.map_err(|_| MemoryError::Invalid("asset filename contains an interior NUL".to_string()))
}

fn io_error(
	operation: &'static str,
	path: impl Into<PathBuf>,
	source: std::io::Error,
) -> MemoryError {
	MemoryError::Io {
		operation,
		path: path.into(),
		source,
	}
}

struct PendingAsset {
	directory: File,
	file: File,
	name: Option<CString>,
	display: PathBuf,
}

impl PendingAsset {
	fn create(directory: &File, directory_path: &Path) -> Result<Self, MemoryError> {
		for _ in 0..32 {
			let name = format!("{TEMP_PREFIX}{}", Uuid::now_v7().hyphenated());
			let name_c = c_name(&name)?;
			// SAFETY: directory is live and generated name is one C component.
			let descriptor = unsafe {
				libc::openat(
					directory.as_raw_fd(),
					name_c.as_ptr(),
					libc::O_RDWR
						| libc::O_CREAT | libc::O_EXCL
						| libc::O_CLOEXEC | libc::O_NOFOLLOW,
					libc::c_uint::from(0o600_u16),
				)
			};
			if descriptor < 0 {
				let source = std::io::Error::last_os_error();
				if source.kind() == std::io::ErrorKind::AlreadyExists {
					continue;
				}
				return Err(io_error(
					"create temporary asset",
					asset_path(directory_path, &name),
					source,
				));
			}
			// SAFETY: successful `openat` returned one newly-owned descriptor.
			let file = unsafe { File::from_raw_fd(descriptor) };
			let display = asset_path(directory_path, &name);
			if let Err(error) = lock_exclusive(&file, &display)
				.and_then(|()| {
					clear_extended_acl(&file)
						.map_err(|source| io_error("clear inherited asset ACL", &display, source))
				})
				.and_then(|()| {
					// SAFETY: this process just created and owns the file.
					if unsafe { libc::fchmod(file.as_raw_fd(), 0o600) } == 0 {
						Ok(())
					} else {
						Err(io_error(
							"secure temporary asset",
							&display,
							std::io::Error::last_os_error(),
						))
					}
				})
				.and_then(|()| validate_asset_metadata(&file, &display, Some(0)).map(drop))
			{
				// SAFETY: directory and generated one-component name are live.
				unsafe {
					libc::unlinkat(directory.as_raw_fd(), name_c.as_ptr(), 0);
				}
				return Err(error);
			}
			let cloned_directory = match directory.try_clone() {
				Ok(cloned) => cloned,
				Err(source) => {
					// SAFETY: directory and generated one-component name are live.
					unsafe {
						libc::unlinkat(directory.as_raw_fd(), name_c.as_ptr(), 0);
					}
					return Err(io_error(
						"duplicate asset directory descriptor",
						directory_path,
						source,
					));
				}
			};
			return Ok(Self {
				directory: cloned_directory,
				file,
				name: Some(name_c),
				display,
			});
		}
		Err(MemoryError::Io {
			operation: "allocate unique temporary asset",
			path: directory_path.to_path_buf(),
			source: std::io::Error::new(
				std::io::ErrorKind::AlreadyExists,
				"temporary asset collision limit reached",
			),
		})
	}

	const fn file(&self) -> &File {
		&self.file
	}

	const fn file_mut(&mut self) -> &mut File {
		&mut self.file
	}

	fn name(&self) -> Option<&CStr> {
		self.name.as_deref()
	}

	fn display(&self) -> &Path {
		&self.display
	}

	fn unlink(&mut self) -> Result<(), MemoryError> {
		let Some(name) = self.name.take() else {
			return Ok(());
		};
		// SAFETY: directory and one-component temporary name are live.
		if unsafe { libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0) } == 0 {
			Ok(())
		} else {
			let source = std::io::Error::last_os_error();
			self.name = Some(name);
			Err(io_error("remove temporary asset", &self.display, source))
		}
	}
}

impl Drop for PendingAsset {
	fn drop(&mut self) {
		if let Some(name) = &self.name {
			// SAFETY: descriptor stays live through this call and the name is
			// one generated component.
			unsafe {
				libc::unlinkat(self.directory.as_raw_fd(), name.as_ptr(), 0);
			}
		}
	}
}

fn lock_exclusive(file: &File, path: &Path) -> Result<(), MemoryError> {
	loop {
		// SAFETY: `file` owns a live descriptor and `LOCK_EX` is valid.
		if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
			return Ok(());
		}
		let source = std::io::Error::last_os_error();
		if source.kind() != std::io::ErrorKind::Interrupted {
			return Err(io_error("lock temporary asset", path, source));
		}
	}
}

fn try_lock_exclusive(file: &File, path: &Path) -> Result<bool, MemoryError> {
	loop {
		// SAFETY: `file` owns a live descriptor and the flock flags are valid.
		if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
			return Ok(true);
		}
		let source = std::io::Error::last_os_error();
		if source.kind() == std::io::ErrorKind::WouldBlock {
			return Ok(false);
		}
		if source.kind() != std::io::ErrorKind::Interrupted {
			return Err(io_error("inspect temporary asset lock", path, source));
		}
	}
}

struct DirectoryStream(*mut libc::DIR);

impl DirectoryStream {
	fn open(directory: &File, path: &Path) -> Result<Self, MemoryError> {
		// SAFETY: directory owns a valid descriptor.
		let descriptor = unsafe { libc::fcntl(directory.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 0) };
		if descriptor < 0 {
			return Err(io_error(
				"duplicate asset directory for enumeration",
				path,
				std::io::Error::last_os_error(),
			));
		}
		// SAFETY: descriptor is an owned duplicate of an open directory.
		let stream = unsafe { libc::fdopendir(descriptor) };
		if stream.is_null() {
			let source = std::io::Error::last_os_error();
			// SAFETY: fdopendir failed and did not take descriptor ownership.
			unsafe {
				libc::close(descriptor);
			}
			return Err(io_error("enumerate asset directory", path, source));
		}
		Ok(Self(stream))
	}

	fn next(&mut self, path: &Path) -> Result<Option<Vec<u8>>, MemoryError> {
		loop {
			clear_errno();
			// SAFETY: stream is live and exclusively borrowed.
			let entry = unsafe { libc::readdir(self.0) };
			if entry.is_null() {
				let errno = current_errno();
				if errno == 0 {
					return Ok(None);
				}
				return Err(io_error(
					"enumerate asset directory",
					path,
					std::io::Error::from_raw_os_error(errno),
				));
			}
			// SAFETY: readdir returned a live dirent with NUL-terminated d_name.
			let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
			if name == b"." || name == b".." {
				continue;
			}
			return Ok(Some(name.to_vec()));
		}
	}
}

impl Drop for DirectoryStream {
	fn drop(&mut self) {
		// SAFETY: this guard owns the DIR stream.
		unsafe {
			libc::closedir(self.0);
		}
	}
}

fn get_extended_acl(file: &File) -> std::io::Result<Option<OwnedAcl>> {
	// SAFETY: descriptor is live and ACL type is valid on macOS.
	let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
	if acl.is_null() {
		let source = std::io::Error::last_os_error();
		if source.raw_os_error() == Some(libc::ENOENT) {
			return Ok(None);
		}
		return Err(source);
	}
	Ok(Some(OwnedAcl(acl)))
}

fn has_extended_acl(file: &File) -> std::io::Result<bool> {
	let Some(acl) = get_extended_acl(file)? else {
		return Ok(false);
	};
	let mut entry: AclEntry = std::ptr::null_mut();
	clear_errno();
	// SAFETY: ACL is live and entry points to writable pointer storage.
	match unsafe { acl_get_entry(acl.0, ACL_FIRST_ENTRY, &raw mut entry) } {
		0 if !entry.is_null() => Ok(true),
		0 => Err(std::io::Error::other(
			"acl_get_entry succeeded without returning an entry",
		)),
		-1 if current_errno() == libc::EINVAL => Ok(false),
		_ => Err(std::io::Error::last_os_error()),
	}
}

fn clear_extended_acl(file: &File) -> std::io::Result<()> {
	let Some(acl) = get_extended_acl(file)? else {
		return Ok(());
	};
	loop {
		let mut entry: AclEntry = std::ptr::null_mut();
		clear_errno();
		// SAFETY: ACL is live and entry points to writable pointer storage.
		match unsafe { acl_get_entry(acl.0, ACL_FIRST_ENTRY, &raw mut entry) } {
			0 if !entry.is_null() => {
				// SAFETY: entry belongs to this live ACL.
				if unsafe { acl_delete_entry(acl.0, entry) } != 0 {
					return Err(std::io::Error::last_os_error());
				}
			}
			0 => {
				return Err(std::io::Error::other(
					"acl_get_entry succeeded without returning an entry",
				));
			}
			-1 if current_errno() == libc::EINVAL => break,
			_ => return Err(std::io::Error::last_os_error()),
		}
	}
	// SAFETY: descriptor and empty ACL are live; type is valid on macOS.
	if unsafe { acl_set_fd_np(file.as_raw_fd(), acl.0, ACL_TYPE_EXTENDED) } != 0 {
		return Err(std::io::Error::last_os_error());
	}
	if has_extended_acl(file)? {
		return Err(std::io::Error::other(
			"extended ACL remained after clearing it",
		));
	}
	Ok(())
}

fn clear_errno() {
	// SAFETY: __error returns this thread's errno pointer on macOS.
	unsafe {
		*libc::__error() = 0;
	}
}

fn current_errno() -> libc::c_int {
	// SAFETY: __error returns this thread's errno pointer on macOS.
	unsafe { *libc::__error() }
}

#[cfg(test)]
#[expect(
	clippy::unwrap_used,
	reason = "asset tests use panic-on-fixture-failure assertions"
)]
mod tests {
	use std::{fs::OpenOptions, io::Write as _, thread};

	use super::*;
	use crate::memory::{SessionEventInput, SessionEventKind};

	fn store() -> (tempfile::TempDir, MemoryStore) {
		let directory = tempfile::tempdir().unwrap();
		let store = MemoryStore::open_path(directory.path().join("emelex.sqlite3")).unwrap();
		(directory, store)
	}

	#[test]
	fn assets_deduplicate_and_tampering_is_detected() {
		let (_directory, store) = store();
		let image = store
			.store_asset_bytes(AssetKind::Image, b"same bytes")
			.unwrap();
		let audio = store
			.store_asset_bytes(AssetKind::Audio, b"same bytes")
			.unwrap();
		assert_eq!(image.sha256(), audio.sha256());
		assert_ne!(image.kind(), audio.kind());
		assert_eq!(store.read_asset(&image).unwrap(), b"same bytes");

		let path = store.assets_dir().join(image.sha256());
		let mut file = OpenOptions::new()
			.write(true)
			.truncate(true)
			.open(path)
			.unwrap();
		file.write_all(b"bad").unwrap();
		file.sync_all().unwrap();
		assert!(matches!(
			store.read_asset(&image),
			Err(MemoryError::Corrupt(_))
		));
	}

	#[test]
	fn gc_preserves_references_then_collects_after_session_deletion() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let asset = store
			.store_asset_bytes(AssetKind::Image, b"linked")
			.unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store.replay_session(&mut lease).unwrap();
		store
			.append_turn(
				&mut lease,
				&[SessionEventInput::new(
					SessionEventKind::UserMessage,
					serde_json::json!({"asset": asset}),
				)
				.with_assets(vec![asset.clone()])],
			)
			.unwrap();
		store.release_session(&lease).unwrap();
		assert_eq!(
			store.gc_assets(Duration::from_nanos(1)).unwrap(),
			AssetGcReport::default()
		);
		store
			.connection()
			.unwrap()
			.execute(
				"DELETE FROM sessions WHERE id = ?1",
				[session.id.to_string()],
			)
			.unwrap();
		let report = store.gc_assets(Duration::from_nanos(1)).unwrap();
		assert_eq!(report.cataloged_files, 1);
		assert!(matches!(
			store.read_asset(&asset),
			Err(MemoryError::Invalid(_))
		));
	}

	#[test]
	fn event_asset_reads_require_exact_link_kind_and_ordinal() {
		let (_directory, store) = store();
		let workspace = tempfile::tempdir().unwrap();
		let session = store.start_session(workspace.path(), None).unwrap();
		let image = store.store_asset_bytes(AssetKind::Image, b"typed").unwrap();
		let mut lease = store.claim_session(session.id, workspace.path()).unwrap();
		store.replay_session(&mut lease).unwrap();
		let events = store
			.append_turn(
				&mut lease,
				&[SessionEventInput::new(
					SessionEventKind::UserMessage,
					serde_json::json!({"asset": image}),
				)
				.with_assets(vec![image.clone()])],
			)
			.unwrap();
		let event = &events[0];
		assert_eq!(
			store.read_event_asset(event.id, 0, &image).unwrap(),
			b"typed"
		);
		let forged_kind =
			AssetRef::new(image.sha256().to_string(), image.bytes(), AssetKind::Audio).unwrap();
		assert!(matches!(
			store.read_event_asset(event.id, 0, &forged_kind),
			Err(MemoryError::Corrupt(_))
		));
		assert!(matches!(
			store.read_event_asset(event.id, 1, &image),
			Err(MemoryError::Corrupt(_))
		));
	}

	#[test]
	fn custom_databases_in_one_parent_have_isolated_asset_namespaces() {
		let directory = tempfile::tempdir().unwrap();
		let first = MemoryStore::open_path(directory.path().join("first.sqlite3")).unwrap();
		let second = MemoryStore::open_path(directory.path().join("second.sqlite3")).unwrap();
		assert_ne!(first.assets_dir(), second.assets_dir());
		let first_asset = first
			.store_asset_bytes(AssetKind::Other, b"shared digest")
			.unwrap();
		let second_asset = second
			.store_asset_bytes(AssetKind::Other, b"shared digest")
			.unwrap();
		assert_eq!(first_asset.sha256(), second_asset.sha256());

		let report = first.gc_assets(Duration::from_nanos(1)).unwrap();
		assert_eq!(report.cataloged_files, 1);
		assert_eq!(second.read_asset(&second_asset).unwrap(), b"shared digest");
	}

	#[test]
	fn concurrent_store_and_gc_leave_a_readable_catalog_entry() {
		let (_directory, store) = store();
		let collector = store.clone();
		let handle = thread::spawn(move || {
			for _ in 0..16 {
				collector.gc_assets(Duration::from_nanos(1)).unwrap();
			}
		});
		let mut latest = None;
		for _ in 0..16 {
			latest = Some(
				store
					.store_asset_bytes(AssetKind::Other, b"concurrent")
					.unwrap(),
			);
		}
		handle.join().unwrap();
		let reference = store
			.store_asset_bytes(AssetKind::Other, b"concurrent")
			.unwrap();
		assert_eq!(latest.unwrap().sha256(), reference.sha256());
		assert_eq!(store.read_asset(&reference).unwrap(), b"concurrent");
	}
}
