from warnings import catch_warnings
import matplotlib.pyplot as plt 
import time
import math
import sys
import random
import matplotlib.cm as cm  # Optional: remove if not needed.
import numpy as np
import heapq

sys.setrecursionlimit(10**7)

# -------------------------------------------------------------
# Configurable parameters
# -------------------------------------------------------------
MAX_ENTRIES = 10         # Maximum entries (M) in a node
MIN_ENTRIES = 4          # Minimum fill (m) in a node (typically ~40% of MAX_ENTRIES)
REINSERT_FACTOR = 0.3    # Fraction of entries to remove when forced reinsertion
SEED = 42                # Random seed for reproducibility
DELAY = 0.1              # Seconds delay after each insertion for visualization
POINT_COUNT = random.randint(150, 200)  # Number of random points

random.seed(SEED)

# -------------------------------------------------------------
# Helper functions: bounding boxes, geometry, etc.
# -------------------------------------------------------------
def create_bounding_box(x, y, x2=None, y2=None):
    """Returns a bounding box in the form (minx, miny, maxx, maxy)."""
    if x2 is None or y2 is None:
        return (x, y, x, y)
    return (min(x, x2), min(y, y2), max(x, x2), max(y, y2))

def union_bounding_boxes(b1, b2):
    return (
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3])
    )

def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def intersection_area(b1, b2):
    ixmin = max(b1[0], b2[0])
    iymin = max(b1[1], b2[1])
    ixmax = min(b1[2], b2[2])
    iymax = min(b1[3], b2[3])
    if ixmax < ixmin or iymax < iymin:
        return 0.0
    return (ixmax - ixmin) * (iymax - iymin)

def total_overlap(bboxes):
    overlap_sum = 0.0
    n = len(bboxes)
    for i in range(n):
        for j in range(i + 1, n):
            overlap_sum += intersection_area(bboxes[i], bboxes[j])
    return overlap_sum

def enlarge(b1, b2):
    new_bb = union_bounding_boxes(b1, b2)
    return area(new_bb) - area(b1)

# -------------------------------------------------------------
# Classes: Entry, Node, RStarTree
# -------------------------------------------------------------
class Entry:
    def __init__(self, bounding_box, child_ptr):
        """
        bounding_box: (minx, miny, maxx, maxy)
        child_ptr:
           - For a leaf node, a data record (x,y)
           - For an internal node, a Node.
        """
        self.bounding_box = bounding_box
        self.child_ptr = child_ptr

class Node:
    def __init__(self, is_leaf=False, level=0):
        self.is_leaf = is_leaf
        self.level = level
        self.entries = []
        self.parent = None

    def bounding_box(self):
        if not self.entries:
            return (0, 0, 0, 0)
        bb = self.entries[0].bounding_box
        for e in self.entries[1:]:
            bb = union_bounding_boxes(bb, e.bounding_box)
        return bb

    @property
    def entry_count(self):
        return len(self.entries)

class RStarTree:
    def __init__(self):
        self.root = Node(is_leaf=True, level=0)
        self.has_forced_reinsert = False

    # ---------------------------------------------------------
    # Searching
    # ---------------------------------------------------------
    def search_nearest(self, x, y):
        """
        Perform a branch-and-bound nearest neighbor search using a priority queue.
        Each node is visited in order of increasing minimum distance from the query point.
        """
        from math import sqrt
        import heapq

        # Compute the Euclidean distance between two points.
        def point_distance(p1, p2):
            return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Compute the minimum distance from a point to a bounding box.
        def distance_to_bb(bb, x, y):
            """
            Compute the minimum distance from a point (x, y) to an axis-aligned bounding box (AABB).
            """
            min_x, min_y, max_x, max_y = bb
    
            # Compute horizontal distance
            dx = 0.0
            if x < min_x:
                dx = min_x - x
            elif x > max_x:
                dx = x - max_x

            # Compute vertical distance
            dy = 0.0
            if y < min_y:
                dy = min_y - y
            elif y > max_y:
                dy = y - max_y

            return (dx**2 + dy**2) ** 0.5  # Euclidean distance


        best_point = None
        best_dist = float('inf')
        heap = []
        iteration_count = 0  # Counter for the number of iterations

        # Push tuple (distance, id(node), node) to ensure a tie-breaker.
        heapq.heappush(heap, (distance_to_bb(self.root.bounding_box(), x, y), id(self.root), self.root))

        while heap:
            iteration_count += 1  # Count the number of times we process a node
            d, _, node = heapq.heappop(heap)
            if d >= best_dist:
                # This node (and all its children) cannot contain a closer point.
                continue
            if node.is_leaf:
                for entry in node.entries:
                    data = entry.child_ptr
                    if isinstance(data, tuple) and len(data) == 2:
                        d_point = point_distance((x, y), data)
                        if d_point < best_dist:
                            best_dist = d_point
                            best_point = data
            else:
                for entry in node.entries:
                    child = entry.child_ptr
                    if isinstance(child, Node):
                        d_child = distance_to_bb(child.bounding_box(), x, y)
                        if d_child < best_dist:
                            heapq.heappush(heap, (d_child, id(child), child))

        print(f"[DEBUG] search_nearest: best distance = {best_dist}, iterations = {iteration_count}")
        return best_point



    # ---------------------------------------------------------
    # Adjust Tree (Bubble-up bounding boxes)
    # ---------------------------------------------------------
    def _adjust_tree(self, node):
        while node is not None:
            bb = None
            for e in node.entries:
                if bb is None:
                    bb = e.bounding_box
                else:
                    bb = union_bounding_boxes(bb, e.bounding_box)
            if node.parent is not None:
                for entry in node.parent.entries:
                    if entry.child_ptr is node:
                        entry.bounding_box = bb
                        break
            node = node.parent

    # ---------------------------------------------------------
    # Insertion
    # ---------------------------------------------------------
    def insert_point(self, x, y):
        self.has_forced_reinsert = False
        entry = Entry(create_bounding_box(x, y), (x, y))
        self._insert_entry(self.root, entry, target_level=0)
        if self.root.entry_count > MAX_ENTRIES:
            self._handle_overflow(self.root)

    def _insert_entry(self, node, entry, target_level, avoid_node_id=None):
        if node is not self.root:
            assert node.parent is not None, f"Node id {id(node)} is not root and has no parent!"
        if node.level == target_level:
            if avoid_node_id is not None and id(node) == avoid_node_id:
                # Avoid inserting into the node from which the entry was removed.
                self._handle_overflow(node)
                new_parent = node.parent if node.parent is not None else self.root
                self._insert_entry(new_parent, entry, target_level, avoid_node_id)
                return
            node.entries.append(entry)
            if isinstance(entry.child_ptr, Node):
                entry.child_ptr.parent = node
                node.is_leaf = False
            if node.entry_count > MAX_ENTRIES:
                self._handle_overflow(node)
            return
        else:
            child_entry = self._choose_subtree(node, entry, avoid_node_id)
            child_entry.bounding_box = union_bounding_boxes(child_entry.bounding_box, entry.bounding_box)
            self._insert_entry(child_entry.child_ptr, entry, target_level, avoid_node_id)
            if node.entry_count > MAX_ENTRIES:
                self._handle_overflow(node)

    def _choose_subtree(self, node, entry, avoid_node_id=None):
        import math
        best_child = None
        min_overlap = math.inf
        min_area_enl = math.inf
        min_area_val = math.inf
        node_bboxes = [c.bounding_box for c in node.entries]
        for c in node.entries:
            if avoid_node_id is not None and id(c.child_ptr) == avoid_node_id:
                continue
            other_bboxes = [bb for bb in node_bboxes if bb is not c.bounding_box]
            current_overlap = total_overlap(other_bboxes)
            new_bb = union_bounding_boxes(c.bounding_box, entry.bounding_box)
            new_overlap = total_overlap(other_bboxes + [new_bb])
            overlap_diff = new_overlap - current_overlap
            if overlap_diff < min_overlap:
                best_child = c
                min_overlap = overlap_diff
                old_area = area(c.bounding_box)
                new_area = area(new_bb)
                min_area_enl = new_area - old_area
                min_area_val = old_area
            elif math.isclose(overlap_diff, min_overlap, rel_tol=1e-12):
                old_area = area(c.bounding_box)
                new_area = area(new_bb)
                area_enl = new_area - old_area
                if area_enl < min_area_enl:
                    best_child = c
                    min_area_enl = area_enl
                    min_area_val = old_area
                elif math.isclose(area_enl, min_area_enl, rel_tol=1e-12):
                    if old_area < min_area_val:
                        best_child = c
                        min_area_val = old_area
        return best_child

    # ---------------------------------------------------------
    # Overflow Handling
    # ---------------------------------------------------------
    def _handle_overflow(self, node):
        print(f"[DEBUG] _handle_overflow on node id {id(node)} (level {node.level}) with {node.entry_count} entries")
        if node is self.root and node.entry_count > MAX_ENTRIES:
            print("[DEBUG] Root overflow => split.")
            self._split(node)
            return
        if node.level == 0:
            print("[DEBUG] Leaf overflow => split.")
            self._split(node)
            return
        if self.has_forced_reinsert:
            print("[DEBUG] Already forced reinsertion => split.")
            self._split(node)
        else:
            print("[DEBUG] Forced reinsertion.")
            self.has_forced_reinsert = True
            self._forced_reinsert(node)
            if node.entry_count > MAX_ENTRIES:
                print("[DEBUG] Still over capacity => split.")
                self._split(node)

    def _forced_reinsert(self, node):
        print(f"[DEBUG] _forced_reinsert on node id {id(node)} (level {node.level}) with {node.entry_count} entries")
        count_to_remove = int(REINSERT_FACTOR * node.entry_count)
        if count_to_remove < 1 and node.entry_count > MAX_ENTRIES:
            count_to_remove = node.entry_count - MAX_ENTRIES
        bb = node.bounding_box()
        cx = 0.5 * (bb[0] + bb[2])
        cy = 0.5 * (bb[1] + bb[3])
        def center_of(bx):
            return (0.5 * (bx[0] + bx[2]), 0.5 * (bx[1] + bx[3]))
        def sqdist(a, b):
            return (a[0] - b[0])**2 + (a[1] - b[1])**2
        node.entries.sort(key=lambda e: sqdist(center_of(e.bounding_box), (cx, cy)), reverse=True)
        reinsert_list = node.entries[:count_to_remove]
        node.entries = node.entries[count_to_remove:]
        for e in reinsert_list:
            if isinstance(e.child_ptr, Node):
                e.child_ptr.parent = None  # detach
            print(f"[DEBUG] Reinserting entry (child id: {id(e.child_ptr)}) into root at level {node.level}, avoiding node id {id(node)}")
            self._insert_entry(self.root, e, target_level=node.level, avoid_node_id=id(node))

    def _split(self, node):
        print(f"[DEBUG] _split on node id {id(node)} (level {node.level}) with {node.entry_count} entries")
        best_split = None
        best_cost = float('inf')
        entries = node.entries[:]
        for dim in [0, 1]:
            entries.sort(key=lambda e: e.bounding_box[dim])
            cand_split, cand_cost = self._try_all_splits(entries)
            if cand_cost < best_cost:
                best_split = cand_split
                best_cost = cand_cost
            entries.sort(key=lambda e: e.bounding_box[dim+2])
            cand_split, cand_cost = self._try_all_splits(entries)
            if cand_cost < best_cost:
                best_split = cand_split
                best_cost = cand_cost
        groupA, groupB = best_split
        newA = Node(is_leaf=node.is_leaf, level=node.level)
        newB = Node(is_leaf=node.is_leaf, level=node.level)
        for e in groupA:
            newA.entries.append(e)
            if not node.is_leaf and isinstance(e.child_ptr, Node):
                e.child_ptr.parent = newA
        for e in groupB:
            newB.entries.append(e)
            if not node.is_leaf and isinstance(e.child_ptr, Node):
                e.child_ptr.parent = newB
        if node is self.root:
            self.root = Node(is_leaf=False, level=node.level + 1)
            self.root.entries = []  # ensure clean slate
            self.root.entries.append(Entry(newA.bounding_box(), newA))
            self.root.entries.append(Entry(newB.bounding_box(), newB))
            newA.parent = self.root
            newB.parent = self.root
        else:
            parent = node.parent if node.parent is not None else self.root
            # Remove *all* references to the old node from parent's entries:
            parent.entries = [pe for pe in parent.entries if pe.child_ptr is not node]
            # Now add entries for newA and newB if not already present:
            if not any(pe.child_ptr is newA for pe in parent.entries):
                parent.entries.append(Entry(newA.bounding_box(), newA))
                newA.parent = parent
            if not any(pe.child_ptr is newB for pe in parent.entries):
                parent.entries.append(Entry(newB.bounding_box(), newB))
                newB.parent = parent
            if parent.entry_count > MAX_ENTRIES:
                self._handle_overflow(parent)
        self._adjust_tree(newA)
        self._adjust_tree(newB)

    def _try_all_splits(self, entries):
        best_partition = None
        best_cost = float('inf')
        n = len(entries)
        for k in range(MIN_ENTRIES, n - MIN_ENTRIES + 1):
            gA = entries[:k]
            gB = entries[k:]
            bbA = gA[0].bounding_box
            for e in gA[1:]:
                bbA = union_bounding_boxes(bbA, e.bounding_box)
            bbB = gB[0].bounding_box
            for e in gB[1:]:
                bbB = union_bounding_boxes(bbB, e.bounding_box)
            cost = area(bbA) + area(bbB)
            if cost < best_cost:
                best_cost = cost
                best_partition = (gA, gB)
        return best_partition, best_cost

    # ---------------------------------------------------------
    # Visualization (using BFS with computed depth)
    # ---------------------------------------------------------
    def visualize(self):
        plt.clf()
        queue = [(self.root, 0)]
        all_boxes = []
        max_depth = 0
        while queue:
            node, depth = queue.pop(0)
            all_boxes.append((depth, node.bounding_box()))
            if not node.is_leaf:
                for e in node.entries:
                    if isinstance(e.child_ptr, Node):
                        queue.append((e.child_ptr, depth+1))
                        max_depth = max(max_depth, depth+1)
        unique_depths = sorted(set(d for d, _ in all_boxes))
        n_depths = len(unique_depths)
        cmap = plt.get_cmap("rainbow", n_depths)
        depth_colors = {d: cmap(i / max(1, n_depths-1)) for i, d in enumerate(unique_depths)}
        for d, bb in all_boxes:
            minx, miny, maxx, maxy = bb
            plt.plot([minx, minx, maxx, maxx, minx],
                     [miny, maxy, maxy, miny, miny],
                     color=depth_colors[d],
                     linewidth=1.5)
        plt.title("R*-Tree Bounding Boxes (colored by depth)")
        plt.axis('equal')
        plt.draw()
        plt.pause(0.001)

# ---------------------------------------------------------
# Print Tree (textual representation with debug info)
# ---------------------------------------------------------
def print_tree(node, level=0, visited=None):
    if visited is None:
        visited = {}
    indent = "    " * level
    if id(node) in visited:
        print(f"{indent}[CYCLE DETECTED] Node id {id(node)} with BB={node.bounding_box()} was first seen at level {visited[id(node)]}.")
        return
    visited[id(node)] = level
    parent_info = f", parent id {id(node.parent)}" if node.parent is not None else ", no parent (root)"
    print(f"{indent}- Node id {id(node)}: Level {node.level} | Entries: {node.entry_count} | is_leaf={node.is_leaf}{parent_info} | BB={node.bounding_box()}")
    for e in node.entries:
        if isinstance(e.child_ptr, Node):
            print_tree(e.child_ptr, level+1, visited)
        else:
            px, py = e.child_ptr
            print(f"{indent}    * Data: ({px:.2f}, {py:.2f})")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(POINT_COUNT)]
    rstar = RStarTree()
    plt.ion()
    fig = plt.figure(figsize=(7, 7))
    for i, (x, y) in enumerate(points, start=1):
        rstar.insert_point(x, y)
        rstar.visualize()
        print(f"Insertion {i}: Inserted ({x:.2f}, {y:.2f})")
    rstar.visualize()
    print_tree(rstar.root)
    ax, ay = zip(*points)
    plt.scatter(ax, ay, c='k', s=10, zorder=5)
    qx, qy = (random.uniform(0, 100), random.uniform(0, 100))
    near = rstar.search_nearest(qx, qy)
    print(f"\nQuery=({qx:.2f}, {qy:.2f}), nearest={near}")
    if near:
        plt.scatter([qx], [qy], c='r', marker='x', s=60, zorder=7)
        plt.scatter([near[0]], [near[1]], c='g', marker='o', s=80, edgecolors='k', zorder=7)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
