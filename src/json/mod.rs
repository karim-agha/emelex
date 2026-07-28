//! Shared fail-closed bounds for caller-controlled JSON trees.

use serde_json::Value;

pub const MAX_DEPTH: usize = 64;
pub const MAX_NODES: usize = 65_536;

/// Check JSON depth and node count without recursively visiting the tree.
///
/// Call this before handing a caller-provided [`Value`] to serde's recursive
/// serializer.
pub fn structurally_bounded(value: &Value) -> bool {
	let mut pending = vec![(value, 0_usize)];
	let mut nodes = 0_usize;
	while let Some((value, depth)) = pending.pop() {
		if depth > MAX_DEPTH {
			return false;
		}
		nodes = nodes.saturating_add(1);
		if nodes > MAX_NODES {
			return false;
		}
		match value {
			Value::Array(values) => {
				if values.len() > MAX_NODES.saturating_sub(nodes.saturating_add(pending.len())) {
					return false;
				}
				pending.extend(values.iter().map(|value| (value, depth + 1)));
			}
			Value::Object(values) => {
				if values.len() > MAX_NODES.saturating_sub(nodes.saturating_add(pending.len())) {
					return false;
				}
				pending.extend(values.values().map(|value| (value, depth + 1)));
			}
			Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
		}
	}
	true
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn accepts_boundary_and_rejects_deeper_tree() {
		let mut boundary = Value::Null;
		for _ in 0..MAX_DEPTH {
			boundary = Value::Array(vec![boundary]);
		}
		assert!(structurally_bounded(&boundary));

		let too_deep = Value::Array(vec![boundary]);
		assert!(!structurally_bounded(&too_deep));
	}

	#[test]
	fn rejects_node_count_above_boundary() {
		let boundary = Value::Array(vec![Value::Null; MAX_NODES - 1]);
		assert!(structurally_bounded(&boundary));

		let too_many = Value::Array(vec![Value::Null; MAX_NODES]);
		assert!(!structurally_bounded(&too_many));
	}
}
