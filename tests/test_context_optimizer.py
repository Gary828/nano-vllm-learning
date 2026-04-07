"""
Tests for ContextOptimizer (ContextPilot integration)

These tests verify that the context reordering and scheduling
optimizations are correctly implemented.
"""

import pytest
import sys
import os

# Add nanovllm directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module file to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "context_optimizer",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "nanovllm", "context_optimizer.py")
)
context_optimizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context_optimizer_module)

IntraContextOrderer = context_optimizer_module.IntraContextOrderer
InterContextScheduler = context_optimizer_module.InterContextScheduler
SimpleHierarchicalClustering = context_optimizer_module.SimpleHierarchicalClustering
ContextOptimizer = context_optimizer_module.ContextOptimizer


class MockNode:
    """Mock node for testing without full tree structure."""

    def __init__(self, node_id, doc_ids=None, original_indices=None, is_leaf=False, is_root=False, children=None, parent=None):
        self.node_id = node_id
        self.doc_ids = doc_ids
        self.original_indices = original_indices or set()
        self.is_leaf = is_leaf
        self.is_root = is_root
        self.children = children or []
        self.parent = parent
        self.frequency = len(self.original_indices)
        self.content = set(doc_ids) if doc_ids else set()


class TestIntraContextOrderer:
    """Tests for IntraContextOrderer."""

    def test_reorder_contexts_simple(self):
        """Test basic context reordering."""
        orderer = IntraContextOrderer()

        # Create simple tree with shared prefix
        root = MockNode(node_id=0, doc_ids=[1, 2, 3], is_root=True, children=[1, 2])
        leaf1 = MockNode(node_id=1, doc_ids=[1, 2, 3, 4, 5], original_indices={0}, is_leaf=True, parent=0)
        leaf2 = MockNode(node_id=2, doc_ids=[1, 2, 3, 6, 7], original_indices={1}, is_leaf=True, parent=0)

        unique_nodes = {0: root, 1: leaf1, 2: leaf2}

        contexts = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]]
        reordered = orderer.reorder_contexts(contexts, unique_nodes)

        # Shared prefix [1, 2, 3] should be at the start
        assert reordered[0][:3] == [1, 2, 3]
        assert reordered[1][:3] == [1, 2, 3]

    def test_reorder_with_parent_prefix(self):
        """Test reordering with parent prefix."""
        orderer = IntraContextOrderer()

        node_docs = [2, 3, 4, 1]
        parent_docs = [1, 2, 3]

        result = orderer._reorder_with_parent_prefix(node_docs, parent_docs)

        # Result should start with parent's prefix
        assert result[:3] == [1, 2, 3]
        assert 4 in result

    def test_find_leaf_node(self):
        """Test finding leaf node for context index."""
        orderer = IntraContextOrderer()

        leaf1 = MockNode(node_id=1, original_indices={0, 1}, is_leaf=True)
        leaf2 = MockNode(node_id=2, original_indices={2}, is_leaf=True)
        unique_nodes = {1: leaf1, 2: leaf2}

        found = orderer._find_leaf_node(0, unique_nodes)
        assert found.node_id == 1

        found = orderer._find_leaf_node(2, unique_nodes)
        assert found.node_id == 2


class TestInterContextScheduler:
    """Tests for InterContextScheduler."""

    def test_group_by_root_prefix(self):
        """Test grouping contexts by root prefix."""
        scheduler = InterContextScheduler()

        search_paths = [
            [0, 0],  # Goes to root's 0th child
            [0, 1],  # Goes to root's 0th child
            [1],     # Goes to root's 1st child
            [],      # No path (should go to -1)
        ]

        groups = scheduler._group_by_root_prefix(search_paths)

        assert groups[0] == [0, 1]  # Both go to child 0
        assert groups[1] == [2]      # Goes to child 1
        assert groups[-1] == [3]     # Empty path

    def test_sort_groups_by_path_length(self):
        """Test sorting groups by path length descending."""
        scheduler = InterContextScheduler()

        groups = {0: [0, 1], 1: [2]}
        search_paths = {
            0: [0, 0, 0],  # length 3
            1: [0, 1],     # length 2
            2: [1],        # length 1
        }
        contexts = [[1, 2], [3, 4], [5, 6]]

        sorted_groups = scheduler._sort_groups_by_path_length(groups, search_paths, contexts)

        # Group [0, 1] should be sorted by path length: 0 (len 3) before 1 (len 2)
        assert sorted_groups[0] == [0, 1]

    def test_schedule_contexts(self):
        """Test full scheduling flow."""
        scheduler = InterContextScheduler()

        reordered = [[1, 2, 3], [1, 2, 4], [5, 6]]
        original = [[1, 2, 3], [1, 2, 4], [5, 6]]
        search_paths = [[0, 0], [0, 1], [1]]

        result_reordered, result_original, order = scheduler.schedule_contexts(
            reordered, original, search_paths
        )

        # [1, 2, x] group should be scheduled before [5, 6]
        assert order[0] in [0, 1]  # First scheduled should be from [1,2,x] group
        assert order[-1] == 2      # [5, 6] should be last


class TestSimpleHierarchicalClustering:
    """Tests for SimpleHierarchicalClustering."""

    def test_fit_transform_single_context(self):
        """Test with single context."""
        clustering = SimpleHierarchicalClustering()

        contexts = [[1, 2, 3, 4, 5]]
        reordered, paths = clustering.fit_transform(contexts)

        assert reordered == [[1, 2, 3, 4, 5]]
        assert paths == [[]]

    def test_fit_transform_identical_contexts(self):
        """Test with identical contexts."""
        clustering = SimpleHierarchicalClustering()

        contexts = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        reordered, paths = clustering.fit_transform(contexts)

        # All contexts are identical, should maintain some ordering
        assert len(reordered) == 3

    def test_shared_prefix_length(self):
        """Test shared prefix length computation."""
        clustering = SimpleHierarchicalClustering()

        # Identical sequences
        assert clustering._shared_prefix_length([1, 2, 3], [1, 2, 3]) == 3

        # Partial match
        assert clustering._shared_prefix_length([1, 2, 3, 4], [1, 2, 3, 5]) == 3

        # No match
        assert clustering._shared_prefix_length([1, 2, 3], [4, 5, 6]) == 0

        # Different lengths
        assert clustering._shared_prefix_length([1, 2, 3], [1, 2]) == 2


class TestContextOptimizer:
    """Tests for the top-level ContextOptimizer."""

    def test_reorder_single_prompt(self):
        """Test with single prompt."""
        optimizer = ContextOptimizer()

        prompts = [[1, 2, 3, 4, 5]]
        reordered, order = optimizer.reorder(prompts)

        assert reordered == [[1, 2, 3, 4, 5]]
        assert order == [0]

    def test_reorder_multiple_prompts(self):
        """Test with multiple prompts having shared prefix."""
        optimizer = ContextOptimizer()

        prompts = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [1, 2, 3, 6, 8],
            [5, 6, 7, 8, 9],
        ]

        reordered, order = optimizer.reorder(prompts)

        # Should have valid reordering
        assert len(reordered) == 4
        assert len(order) == 4

        # Shared prefix [1, 2, 3] should be grouped together
        # Find indices of [1,2,3,x] group in reordered
        group_indices = [i for i, ctx in enumerate(reordered) if ctx[:3] == [1, 2, 3]]
        other_indices = [i for i, ctx in enumerate(reordered) if ctx[:2] == [5, 6]]

        # [1,2,3] group should be scheduled before [5,6] group
        if group_indices and other_indices:
            assert max(group_indices) < min(other_indices)

    def test_reorder_no_optimization_single(self):
        """Test that single prompt is returned as-is."""
        optimizer = ContextOptimizer()

        prompts = [[1, 2, 3]]
        reordered, order = optimizer.reorder(prompts)

        assert reordered == [[1, 2, 3]]
        assert order == [0]

    def test_reorder_empty_list(self):
        """Test with empty list."""
        optimizer = ContextOptimizer()

        prompts = []
        reordered, order = optimizer.reorder(prompts)

        assert reordered == []
        assert order == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
