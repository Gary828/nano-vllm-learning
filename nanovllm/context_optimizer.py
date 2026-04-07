"""
ContextPilot 集成模块：用于优化 KV Cache 前缀共享

nano-vllm 是一个轻量级 vLLM 实现，此模块将 ContextPilot 的
上下文重排序技术集成进来，以提升前缀缓存命中率。

核心功能：
1. Intra-context 重排序：将 prompt 内共享前缀多的 tokens 放前面
2. Inter-context 调度：批量调度时，长公共前缀的序列优先处理

参考 ContextPilot:
- https://github.com/EfficientContext/ContextPilot
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np


class IntraContextOrderer:
    """
    单个 context 内部的重排序器。

    基于聚类树结构，对单个 context 内部的元素重排序，
    使共享前缀最大化，提升 KV cache 命中率。

    简化实现（不依赖 scipy）：使用前缀树 + 频率统计。
    """

    def reorder_contexts(
        self,
        original_contexts: List[List[int]],
        unique_nodes: Dict[int, Any],
    ) -> List[List[int]]:
        """
        基于聚类树结构重排序 contexts。

        Args:
            original_contexts: List of original context lists (each is list of token IDs)
            unique_nodes: Dictionary of unique tree nodes from clustering

        Returns:
            List of reordered contexts with optimized prefix sharing
        """
        if not original_contexts:
            return original_contexts

        # Find root node
        root_node = None
        for node_id, node in unique_nodes.items():
            if getattr(node, "is_root", False):
                root_node = node
                break

        if not root_node:
            return original_contexts

        # Assign original contexts to leaf nodes
        for node_id, node in unique_nodes.items():
            if getattr(node, "is_leaf", False) and hasattr(node, "original_indices"):
                orig_indices = getattr(node, "original_indices", set())
                if orig_indices:
                    first_idx = min(orig_indices)
                    if first_idx < len(original_contexts):
                        node.doc_ids = list(original_contexts[first_idx])

        # Top-down traversal: reorder each node to start with parent's prefix
        from collections import deque

        queue = deque([root_node.node_id])
        visited = set()

        while queue:
            node_id = queue.popleft()
            if node_id in visited or node_id not in unique_nodes:
                continue
            visited.add(node_id)

            node = unique_nodes[node_id]

            # If not root and has parent, reorder to start with parent's prefix
            if not getattr(node, "is_root", False) and getattr(node, "parent", None) is not None:
                parent_node = unique_nodes.get(getattr(node, "parent", None))
                if parent_node and getattr(parent_node, "doc_ids", None) and getattr(node, "doc_ids", None):
                    node.doc_ids = self._reorder_with_parent_prefix(
                        node.doc_ids,
                        parent_node.doc_ids,
                    )

            # Add children to queue
            children = getattr(node, "children", None)
            if children:
                for child_id in children:
                    if child_id in unique_nodes:
                        queue.append(child_id)

        # Extract reordered contexts from leaf nodes
        reordered_contexts = []
        for i, original_context in enumerate(original_contexts):
            leaf_node = self._find_leaf_node(i, unique_nodes)
            if leaf_node and getattr(leaf_node, "doc_ids", None):
                reordered_contexts.append(list(leaf_node.doc_ids))
            else:
                reordered_contexts.append(list(original_context))

        return reordered_contexts

    def _reorder_with_parent_prefix(
        self,
        node_docs: List[int],
        parent_docs: List[int],
    ) -> List[int]:
        """
        Reorder a node's documents to start with the parent's prefix.

        Example: node=[2,3,4,1], parent=[1,2,3] -> result=[1,2,3,4]
        """
        if not parent_docs:
            return node_docs

        result = list(parent_docs)
        parent_set = set(parent_docs)
        for doc in node_docs:
            if doc not in parent_set:
                result.append(doc)

        return result

    def _find_leaf_node(self, context_index: int, unique_nodes: Dict[int, Any]):
        """Find the leaf node that contains the given context index."""
        for node in unique_nodes.values():
            if getattr(node, "is_leaf", False):
                orig_indices = getattr(node, "original_indices", set())
                if context_index in orig_indices:
                    return node
        return None

    def extract_search_paths(
        self,
        unique_nodes: Dict[int, Any],
        num_contexts: int,
    ) -> List[List[int]]:
        """
        Extract search paths (child indices) for each context.

        Returns:
            List of search paths, where each path is a list of child indices
        """
        search_paths = [[] for _ in range(num_contexts)]

        # Build a mapping from context index to its leaf node
        context_to_leaf = {}
        for node_id, node in unique_nodes.items():
            if getattr(node, "is_leaf", False):
                orig_indices = getattr(node, "original_indices", set())
                for orig_idx in orig_indices:
                    context_to_leaf[orig_idx] = node_id

        # Extract search paths for each context
        for context_idx in range(num_contexts):
            if context_idx not in context_to_leaf:
                search_paths[context_idx] = []
                continue

            # Trace upward from leaf to root, recording child indices
            child_indices = []
            current_id = context_to_leaf[context_idx]
            visited = set()

            while current_id is not None and current_id in unique_nodes:
                if current_id in visited:
                    break
                visited.add(current_id)

                current_node = unique_nodes[current_id]
                parent_id = getattr(current_node, "parent", None)

                if parent_id is not None and parent_id in unique_nodes:
                    parent_node = unique_nodes[parent_id]
                    try:
                        children = getattr(parent_node, "children", [])
                        child_index = children.index(current_id)
                        child_indices.append(child_index)
                    except (ValueError, AttributeError):
                        pass

                current_id = parent_id

            # Reverse to get root-to-leaf path of child indices
            search_paths[context_idx] = child_indices[::-1]

        return search_paths


class InterContextScheduler:
    """
    批量调度优化器。

    O(N) 分组 + O(N log N) 排序
    基于 ContextPilot 的 InterContextScheduler 算法。

    核心思想：
    1. 按 search_path[0] 分组，自然分离 cache 区域
    2. 组内按 path_length 降序排序（长前缀优先）
    """

    def schedule_contexts(
        self,
        reordered_contexts: List[List[int]],
        original_contexts: List[List[int]],
        search_paths: List[List[int]],
    ) -> Tuple[List[List[int]], List[List[int]], List[int]]:
        """
        Schedule contexts using search-path-based grouping and sorting.

        Args:
            reordered_contexts: Reordered contexts from IntraContextOrderer
            original_contexts: Original contexts
            search_paths: Search paths from clustering

        Returns:
            (scheduled_reordered, scheduled_originals, final_index_mapping)
        """
        # Step 1: Group contexts by the first child index in their search path
        # O(N) complexity
        groups_by_root = self._group_by_root_prefix(search_paths)

        # Step 2: Sort by path length in descending order within each group
        sorted_groups = self._sort_groups_by_path_length(
            groups_by_root, search_paths, reordered_contexts
        )

        # Step 3: Create final ordering
        # Sort groups by size (largest first), then by first index for deterministic ordering
        all_groups_sorted = []
        for group_indices in sorted_groups:
            all_groups_sorted.append(group_indices)

        all_groups_sorted.sort(
            key=lambda x: (-len(x), x[0] if x else float("inf"))
        )

        final_index_mapping = [idx for group in all_groups_sorted for idx in group]

        scheduled_reordered = [reordered_contexts[i] for i in final_index_mapping]
        scheduled_originals = [original_contexts[i] for i in final_index_mapping]

        return scheduled_reordered, scheduled_originals, final_index_mapping

    def _group_by_root_prefix(
        self,
        search_paths: List[List[int]],
    ) -> Dict[int, List[int]]:
        """
        Group contexts by the first child index in their search path.

        O(N) operation that naturally separates contexts into cache regions.
        """
        groups = defaultdict(list)

        for context_idx, path in enumerate(search_paths):
            if len(path) >= 1:
                group_key = path[0]
                groups[group_key].append(context_idx)
            else:
                groups[-1].append(context_idx)

        return groups

    def _sort_groups_by_path_length(
        self,
        groups_by_root: Dict[int, List[int]],
        search_paths: List[List[int]],
        contexts: List[List[int]],
    ) -> List[List[int]]:
        """
        Sort contexts within each group by path length descending,
        with lexicographic tiebreaker for equal-length paths.
        """
        sorted_groups = []

        for root_prefix, group_indices in groups_by_root.items():
            sorted_group = sorted(
                group_indices,
                key=lambda idx: (-len(search_paths[idx]), search_paths[idx], idx),
            )
            sorted_groups.append(sorted_group)

        return sorted_groups


class SimpleHierarchicalClustering:
    """
    简化版层次聚类（不依赖 scipy）。

    基于共享前缀长度和频率的简单排序算法。
    核心思想：让共享前缀多的序列连续处理，提高 KV cache 命中率。

    时间复杂度：O(N² * L) 其中 L 是平均序列长度。
    """

    def fit_transform(self, contexts: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Perform simplified clustering and return reordered contexts.

        Args:
            contexts: List of token sequences

        Returns:
            (reordered_contexts, search_paths)
        """
        n = len(contexts)
        if n < 2:
            return contexts, [[] for _ in contexts]

        # Compute shared prefix lengths for all pairs
        prefix_lengths = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j:
                    prefix_lengths[i, j] = self._shared_prefix_length(contexts[i], contexts[j])

        # Group contexts by their longest shared prefix with others
        # Higher shared prefix = should be processed together
        groups = self._group_by_prefix_affinity(contexts, prefix_lengths)

        # Build tree structure from groups
        tree = self._build_tree(contexts, groups)

        # Reorder contexts using the tree
        intra_orderer = IntraContextOrderer()
        reordered = intra_orderer.reorder_contexts(contexts, tree)
        search_paths = intra_orderer.extract_search_paths(tree, n)

        return reordered, search_paths

    def _shared_prefix_length(self, a: List[int], b: List[int]) -> int:
        """Compute the length of shared prefix between two sequences."""
        length = 0
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                length += 1
            else:
                break
        return length

    def _group_by_prefix_affinity(
        self, contexts: List[List[int]], prefix_lengths: np.ndarray
    ) -> List[List[int]]:
        """
        Group contexts by prefix affinity.

        Returns groups where contexts within each group share long prefixes.
        """
        n = len(contexts)
        used = [False] * n
        groups = []

        for start_i in range(n):
            if used[start_i]:
                continue

            # Start a new group with context start_i
            group = [start_i]
            used[start_i] = True

            # Find all contexts that share significant prefix with this group
            for _ in range(n - start_i - 1):
                best_j = -1
                best_prefix = 0

                for j in range(n):
                    if used[j]:
                        continue

                    # Check prefix length with first member in group
                    if prefix_lengths[group[0]][j] >= 2:  # Threshold: at least 2 shared tokens
                        if best_j == -1 or prefix_lengths[group[0]][j] > best_prefix:
                            best_j = j
                            best_prefix = prefix_lengths[group[0]][j]

                if best_j != -1:
                    group.append(best_j)
                    used[best_j] = True
                else:
                    break

            groups.append(group)

        return groups

    def _build_tree(self, contexts: List[List[int]], groups: List[List[int]]) -> Dict[int, Any]:
        """
        Build a simplified tree structure from groups.

        Returns:
            Dictionary mapping node_id to node objects
        """
        n = len(contexts)

        class SimpleNode:
            def __init__(self, node_id, content, original_indices, is_leaf=False, is_root=False):
                self.node_id = node_id
                self.content = content  # set of token IDs
                self.original_indices = original_indices
                self.doc_ids = None
                self.is_leaf = is_leaf
                self.is_root = is_root
                self.children = []
                self.parent = None
                self.frequency = len(original_indices)

        nodes = {}
        next_id = n

        # Create leaf nodes
        for i, ctx in enumerate(contexts):
            nodes[i] = SimpleNode(i, set(ctx), {i}, is_leaf=True)

        # Create root
        root_id = next_id
        root_content = set()
        root_indices = set()
        for group in groups:
            for idx in group:
                root_content |= set(contexts[idx])
                root_indices.add(idx)
        nodes[root_id] = SimpleNode(root_id, root_content, root_indices, is_root=True)
        next_id += 1

        # Link groups to root
        for group in groups:
            if len(group) > 1:
                # Create internal node for group
                group_id = next_id
                group_content = set()
                group_indices = set()
                for idx in group:
                    group_content |= set(contexts[idx])
                    group_indices.add(idx)
                nodes[group_id] = SimpleNode(group_id, group_content, group_indices)
                nodes[group_id].children = group
                nodes[group_id].parent = root_id
                nodes[root_id].children.append(group_id)

                for idx in group:
                    nodes[idx].parent = group_id
            else:
                nodes[group[0]].parent = root_id
                nodes[root_id].children.append(group[0])

        return nodes


class ContextOptimizer:
    """
    顶层上下文优化器。

    封装 ContextPilot 核心算法：
    1. 简化版层次聚类 (SimpleHierarchicalClustering)
    2. Intra-context 重排序 (IntraContextOrderer)
    3. Inter-context 调度 (InterContextScheduler)
    """

    def __init__(self, use_scipy: bool = False):
        """
        Initialize ContextOptimizer.

        Args:
            use_scipy: Whether to use scipy for clustering (not implemented in nano-vllm)
        """
        self.clustering = SimpleHierarchicalClustering()
        self.intra_orderer = IntraContextOrderer()
        self.inter_scheduler = InterContextScheduler()

    def reorder(self, prompts: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """
        主入口：对一批 prompts 进行重排序和调度优化。

        Args:
            prompts: List of token sequences

        Returns:
            (reordered_prompts, execution_order)
        """
        n = len(prompts)
        if n < 2:
            return prompts, list(range(n))

        # Step 1: Hierarchical clustering to build tree structure
        reordered_contexts, search_paths = self.clustering.fit_transform(prompts)

        # Step 2: Inter-context scheduling
        (
            scheduled_reordered,
            _,
            final_order,
        ) = self.inter_scheduler.schedule_contexts(
            reordered_contexts, prompts, search_paths
        )

        return scheduled_reordered, final_order
