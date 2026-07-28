use std::{
	os::unix::fs::{PermissionsExt as _, symlink},
	process::Command,
};

use super::*;

#[test]
fn explicit_home_has_highest_precedence() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("explicit");
	let home = EmelexHome::resolve(Some(&selected)).expect("home resolves");
	assert_eq!(
		home.root(),
		fs::canonicalize(selected)
			.expect("explicit home exists")
			.as_path()
	);
}

#[test]
fn empty_environment_paths_are_not_relative_storage_roots() {
	assert_eq!(
		requested_root(
			None,
			Some(OsString::new()),
			Some(OsString::from("/tmp/example-home"))
		)
		.expect("empty override falls through"),
		PathBuf::from("/tmp/example-home/.emelex")
	);
	assert!(matches!(
		requested_root(None, Some(OsString::new()), Some(OsString::new())),
		Err(HomeError::HomeUnavailable)
	));
}

#[test]
fn prepared_home_contains_standard_directories() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("emelex")).expect("home prepares");
	assert!(home.models_dir().is_dir());
	let marker = fs::metadata(home.root().join(HOME_MARKER_NAME)).expect("marker metadata");
	assert_eq!(marker.permissions().mode() & 0o777, 0o600);
}

#[test]
fn prepared_home_rejects_symlinked_standard_directory() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("emelex");
	let outside = temp.path().join("outside");
	fs::create_dir(&outside).expect("outside directory");
	EmelexHome::prepare(&selected).expect("initial home");
	fs::remove_dir(selected.join("cache")).expect("remove owned cache");
	symlink(&outside, selected.join("cache")).expect("cache symlink");
	let error = EmelexHome::prepare(&selected).expect_err("symlink must be rejected");
	assert!(error.to_string().contains("cache"));
}

#[test]
fn filesystem_root_is_rejected_without_permission_changes() {
	let before = fs::metadata("/")
		.expect("root metadata")
		.permissions()
		.mode();
	let error = EmelexHome::prepare(Path::new("/")).expect_err("root must be rejected");
	let after = fs::metadata("/")
		.expect("root metadata after rejection")
		.permissions()
		.mode();

	assert!(matches!(error, HomeError::UnsafePath { .. }));
	assert_eq!(before, after);
}

#[test]
fn shared_existing_directory_is_rejected_without_mutation() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("shared");
	fs::create_dir(&selected).expect("shared directory");
	fs::set_permissions(&selected, fs::Permissions::from_mode(0o755))
		.expect("set shared permissions");
	fs::write(selected.join("keep.txt"), "keep").expect("write shared content");

	let error = EmelexHome::prepare(&selected).expect_err("shared directory must be rejected");

	assert!(matches!(error, HomeError::UnsafePath { .. }));
	assert_eq!(
		fs::metadata(&selected)
			.expect("shared metadata")
			.permissions()
			.mode() & 0o777,
		0o755
	);
	assert_eq!(
		fs::read_to_string(selected.join("keep.txt")).expect("shared content"),
		"keep"
	);
	assert!(!selected.join(HOME_MARKER_NAME).exists());
	assert!(!selected.join("cache").exists());
}

#[test]
fn owner_only_nonempty_unmarked_directory_is_not_adopted() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("unmarked");
	fs::create_dir(&selected).expect("unmarked directory");
	fs::set_permissions(&selected, fs::Permissions::from_mode(0o700))
		.expect("set owner-only permissions");
	fs::write(selected.join("foreign"), "data").expect("write foreign content");

	let error = EmelexHome::prepare(&selected).expect_err("unmarked directory must be rejected");

	assert!(matches!(error, HomeError::UnsafePath { .. }));
	assert!(!selected.join(HOME_MARKER_NAME).exists());
	assert!(!selected.join("cache").exists());
}

#[test]
fn empty_owner_only_directory_can_be_adopted_once() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("empty");
	fs::create_dir(&selected).expect("empty directory");
	fs::set_permissions(&selected, fs::Permissions::from_mode(0o700))
		.expect("set owner-only permissions");

	let first = EmelexHome::prepare(&selected).expect("empty directory is adopted");
	let second = EmelexHome::prepare(&selected).expect("marked directory reopens");

	assert_eq!(first, second);
	assert_eq!(
		fs::read(first.root().join(HOME_MARKER_NAME)).expect("read marker"),
		HOME_MARKER_CONTENT
	);
}

#[test]
fn final_home_symlink_is_rejected_without_touching_target() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let outside = temp.path().join("outside");
	let selected = temp.path().join("selected");
	fs::create_dir(&outside).expect("outside directory");
	fs::set_permissions(&outside, fs::Permissions::from_mode(0o755)).expect("outside permissions");
	symlink(&outside, &selected).expect("home symlink");

	assert!(EmelexHome::prepare(&selected).is_err());
	assert_eq!(
		fs::metadata(&outside)
			.expect("outside metadata")
			.permissions()
			.mode() & 0o777,
		0o755
	);
	assert!(!outside.join(HOME_MARKER_NAME).exists());
}

#[test]
fn existing_home_with_extended_acl_is_rejected_without_mutation() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("home");
	EmelexHome::prepare(&selected).expect("initial home");
	add_read_acl(&selected, false);
	let before = acl_listing(&selected);

	let error = EmelexHome::prepare(&selected).expect_err("ACL home must be rejected");

	assert!(matches!(error, HomeError::UnsafePath { .. }));
	assert_eq!(acl_listing(&selected), before);
}

#[test]
fn existing_owned_child_with_extended_acl_is_rejected() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let selected = temp.path().join("home");
	EmelexHome::prepare(&selected).expect("initial home");
	add_read_acl(&selected.join("cache"), false);

	let error = EmelexHome::prepare(&selected).expect_err("ACL child must be rejected");

	assert!(error.to_string().contains("cache"));
	assert!(has_acl_entry(&acl_listing(&selected.join("cache"))));
}

#[test]
fn new_home_clears_inherited_extended_acls() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let parent = temp.path().join("parent");
	let selected = parent.join("home");
	fs::create_dir(&parent).expect("create parent");
	add_read_acl(&parent, true);

	let home = EmelexHome::prepare(&selected).expect("home below ACL parent");

	assert!(!has_acl_entry(&acl_listing(home.root())));
	assert!(!has_acl_entry(&acl_listing(&home.cache_dir())));
	assert!(!has_acl_entry(&acl_listing(
		&home.root().join(HOME_MARKER_NAME)
	)));
}

#[test]
fn snapshot_mutation_lock_can_be_polled_without_blocking() {
	let temp = tempfile::tempdir().expect("temporary directory");
	let home = EmelexHome::prepare(&temp.path().join("home")).expect("home");
	let first = home.lock_snapshot_mutations().expect("first lock");
	assert!(
		home.try_lock_snapshot_mutations()
			.expect("nonblocking lock probe")
			.is_none()
	);
	drop(first);
	assert!(
		home.try_lock_snapshot_mutations()
			.expect("released lock probe")
			.is_some()
	);
}

fn add_read_acl(path: &Path, inheritable: bool) {
	let rule = if inheritable {
		"everyone allow list,search,readattr,readextattr,readsecurity,file_inherit,directory_inherit"
	} else {
		"everyone allow list,search,readattr,readextattr,readsecurity"
	};
	let output = Command::new("/bin/chmod")
		.args(["+a", rule])
		.arg(path)
		.output()
		.expect("run chmod ACL");
	assert!(
		output.status.success(),
		"chmod ACL failed: {}",
		String::from_utf8_lossy(&output.stderr)
	);
}

fn acl_listing(path: &Path) -> String {
	let output = Command::new("/bin/ls")
		.arg("-lde")
		.arg(path)
		.output()
		.expect("list ACL");
	assert!(output.status.success());
	String::from_utf8_lossy(&output.stdout).into_owned()
}

fn has_acl_entry(listing: &str) -> bool {
	listing
		.lines()
		.skip(1)
		.any(|line| line.trim_start().starts_with("0:"))
}
