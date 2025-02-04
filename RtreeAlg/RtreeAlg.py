import matplotlib.pyplot as plt
import time
import math
import sys
import random

sys.setrecursionlimit(10**7)

# -------------------------------------------------------------
# Configurable parameters
# -------------------------------------------------------------
MAX_ENTRIES = 6          # Maximum entries (M) in a node
MIN_ENTRIES = 3          # Minimum fill (m) in a node
REINSERT_FACTOR = 0.3    # Fraction of entries to remove when forced reinsert
SEED = 42                # Random seed for reproducibility
DELAY = 0.1              # Seconds delay after each insertion for visualization
POINT_COUNT = random.randint(30, 50)  # Number of random points

random.seed(SEED)

# -------------------------------------------------------------
# Helper functions: bounding boxes, geometry, etc.
# -------------------------------------------------------------
def create_bounding_box(x, y, x2=None, y2=None):
    """Returns a bounding box in the form (minx, miny, maxx, maxy)."""
    if x2 is None or y2 is None:
        return (x, y, x, y)  # single-point bbox
    return (
        min(x, x2),
        min(y, y2),
        max(x, x2),
        max(y, y2),
    )

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
            return (0,0,0,0)
        b = self.entries[0].bounding_box
        for e in self.entries[1:]:
            b = union_bounding_boxes(b, e.bounding_box)
        return b

    @property
    def entry_count(self):
        return len(self.entries)

class RStarTree:
    def __init__(self):
        self.root = Node(is_leaf=True, level=0)
        self.has_forced_reinsert = False  # track if forced reinsert used this insertion

    def insert_point(self, x, y):
        """Insert a single 2D point (x,y) into the R*-Tree."""
        self.has_forced_reinsert = False  # reset for each new insertion

        entry = Entry(create_bounding_box(x,y), (x,y))
        self._insert_entry(self.root, entry)

        if self.root.entry_count > MAX_ENTRIES:
            self._handle_overflow(self.root)

    def _insert_entry(self, node, entry):
        if node.is_leaf:
            node.entries.append(entry)
        else:
            child_entry = self._choose_subtree(node, entry)
            child_entry.bounding_box = union_bounding_boxes(child_entry.bounding_box, entry.bounding_box)
            self._insert_entry(child_entry.child_ptr, entry)

        if node.entry_count > MAX_ENTRIES:
            self._handle_overflow(node)

    def _choose_subtree(self, node, entry):
        import math
        best_child = None
        min_overlap_enlargement = math.inf
        min_area_enlargement = math.inf
        min_area_value = math.inf

        node_bboxes = [c.bounding_box for c in node.entries]
        for c in node.entries:
            other_bboxes = [bb for bb in node_bboxes if bb is not c.bounding_box]
            current_overlap = total_overlap(other_bboxes)

            new_bb = union_bounding_boxes(c.bounding_box, entry.bounding_box)
            new_overlap = total_overlap(other_bboxes + [new_bb])
            overlap_diff = new_overlap - current_overlap

            if overlap_diff < min_overlap_enlargement:
                best_child = c
                min_overlap_enlargement = overlap_diff
                old_area = area(c.bounding_box)
                new_area = area(new_bb)
                min_area_enlargement = new_area - old_area
                min_area_value = old_area
            elif math.isclose(overlap_diff, min_overlap_enlargement, rel_tol=1e-12):
                old_area = area(c.bounding_box)
                new_area = area(new_bb)
                area_enl = new_area - old_area
                if area_enl < min_area_enlargement:
                    best_child = c
                    min_area_enlargement = area_enl
                    min_area_value = old_area
                elif math.isclose(area_enl, min_area_enlargement, rel_tol=1e-12):
                    if old_area < min_area_value:
                        best_child = c
                        min_area_value = old_area
        return best_child

    def _handle_overflow(self, node):
        print(f"[DEBUG] _handle_overflow on node@{id(node)} level={node.level}, entry_count={node.entry_count}")

        # 1) If node is root => always split
        if node is self.root:
            print("[DEBUG] Root is over capacity => split (no forced reinsert).")
            self._split(node)
            return

        # 2) If node is a leaf (level=0) => split immediately
        if node.level == 0:
            print("[DEBUG] Leaf node is over capacity => split (no forced reinsert).")
            self._split(node)
            return

        # 3) Non-leaf, non-root => possibly forced reinsert once
        if self.has_forced_reinsert:
            print("[DEBUG] Already used forced reinsert this insertion => split.")
            self._split(node)
        else:
            print("[DEBUG] Forced reinsert on internal node (non-root, non-leaf).")
            self._forced_reinsert(node)
            self.has_forced_reinsert = True

            if node.entry_count > MAX_ENTRIES:
                print("[DEBUG] Still over capacity after forced reinsert => split.")
                self._split(node)

    def _forced_reinsert(self, node):
        print(f"[DEBUG] _forced_reinsert on node@{id(node)} level={node.level}, entry_count={node.entry_count}")

        count_to_remove = int(REINSERT_FACTOR * node.entry_count)
        if count_to_remove < 1 and node.entry_count > MAX_ENTRIES:
            count_to_remove = node.entry_count - MAX_ENTRIES

        node_bb = node.bounding_box()
        cx = 0.5 * (node_bb[0] + node_bb[2])
        cy = 0.5 * (node_bb[1] + node_bb[3])

        def center_of(bb):
            return (0.5*(bb[0]+bb[2]), 0.5*(bb[1]+bb[3]))
        def dist(a, b):
            return (a[0]-b[0])**2 + (a[1]-b[1])**2

        node.entries.sort(
            key=lambda e: dist(center_of(e.bounding_box), (cx, cy)),
            reverse=True
        )

        reinsert_list = node.entries[:count_to_remove]
        node.entries = node.entries[count_to_remove:]

        print(f"[DEBUG] Removing {count_to_remove} entries for reinsert.")

        for e in reinsert_list:
            print("[DEBUG] Reinserting entry into root.")
            self._insert_entry(self.root, e)

    def _split(self, node):
        print(f"[DEBUG] _split on node at level {node.level}, entry_count={node.entry_count}")
        import math
        best_split = None
        best_criteria = math.inf

        entries = node.entries[:]
        dim_choices = [0, 1]

        for dim in dim_choices:
            entries.sort(key=lambda e: e.bounding_box[dim])
            cand_split, cand_val = self._try_all_splits(entries)
            if cand_val < best_criteria:
                best_split = cand_split
                best_criteria = cand_val

            entries.sort(key=lambda e: e.bounding_box[dim+2])
            cand_split, cand_val = self._try_all_splits(entries)
            if cand_val < best_criteria:
                best_split = cand_split
                best_criteria = cand_val

        groupA, groupB = best_split

        newNodeA = Node(is_leaf=node.is_leaf, level=node.level)
        newNodeB = Node(is_leaf=node.is_leaf, level=node.level)

        for e in groupA:
            newNodeA.entries.append(e)
            if not node.is_leaf:
                e.child_ptr.parent = newNodeA
        for e in groupB:
            newNodeB.entries.append(e)
            if not node.is_leaf:
                e.child_ptr.parent = newNodeB

        if node is self.root:
            self.root = Node(is_leaf=False, level=node.level + 1)
            self.root.entries.append(Entry(newNodeA.bounding_box(), newNodeA))
            self.root.entries.append(Entry(newNodeB.bounding_box(), newNodeB))
            newNodeA.parent = self.root
            newNodeB.parent = self.root
        else:
            parent = node.parent
            for i, pe in enumerate(parent.entries):
                if pe.child_ptr is node:
                    parent.entries[i] = Entry(newNodeA.bounding_box(), newNodeA)
                    break
            parent.entries.append(Entry(newNodeB.bounding_box(), newNodeB))
            newNodeA.parent = parent
            newNodeB.parent = parent

            if parent.entry_count > MAX_ENTRIES:
                self._handle_overflow(parent)

    def _try_all_splits(self, sorted_entries):
        best_partition = None
        best_cost = math.inf

        n = len(sorted_entries)
        min_split = MIN_ENTRIES
        max_split = n - MIN_ENTRIES

        for k in range(min_split, max_split + 1):
            groupA = sorted_entries[:k]
            groupB = sorted_entries[k:]

            bbA = groupA[0].bounding_box
            for e in groupA[1:]:
                bbA = union_bounding_boxes(bbA, e.bounding_box)

            bbB = groupB[0].bounding_box
            for e in groupB[1:]:
                bbB = union_bounding_boxes(bbB, e.bounding_box)

            cost = area(bbA) + area(bbB)
            if cost < best_cost:
                best_cost = cost
                best_partition = (groupA, groupB)

        return best_partition, best_cost

    # ---------------------------
    # Visualization
    # ---------------------------
    def visualize(self):
        """
        Plot bounding boxes of all nodes (without points).
        We do a full redraw:
          1) plt.clf()
          2) draw bounding boxes
          3) plt.draw() / plt.pause()
        """
        plt.clf()
        stack = [(self.root, self.root.level)]
        max_level = self.root.level
        boxes_by_level = {}

        while stack:
            node, lvl = stack.pop()
            if lvl not in boxes_by_level:
                boxes_by_level[lvl] = []
            boxes_by_level[lvl].append(node.bounding_box())

            if not node.is_leaf:
                for entry in node.entries:
                    child = entry.child_ptr
                    if isinstance(child, Node):
                        stack.append((child, child.level))
                        if child.level > max_level:
                            max_level = child.level

        color_map = plt.get_cmap('hsv', max_level + 2)

        # Draw bounding boxes by level
        for lvl in sorted(boxes_by_level.keys()):
            for b in boxes_by_level[lvl]:
                (minx, miny, maxx, maxy) = b
                plt.plot(
                    [minx, minx, maxx, maxx, minx],
                    [miny, maxy, maxy, miny, miny],
                    color=color_map(lvl)
                )

        plt.title(f"R*-Tree bounding boxes (colored by level)")
        plt.axis('equal')
        plt.draw()
        plt.pause(0.001)

# -------------------------------------------------------------
# Main: Generate points, build, and visualize
# -------------------------------------------------------------
def main():
    # 1) Generate all points
    points = [(random.uniform(0,100), random.uniform(0,100)) for _ in range(POINT_COUNT)]
    # 2) Create the R*-Tree
    rstar = RStarTree()

    # Turn on interactive plotting
    plt.ion()
    fig = plt.figure(figsize=(7,7))

    # 3) Insert each point, visualize the bounding boxes (NO points)
    for i, (x,y) in enumerate(points, start=10):
        rstar.insert_point(x, y)
        # draw bounding boxes each step
        rstar.visualize()
        print(f"Insertion {i}: Inserted point ({x:.2f}, {y:.2f})")

    # 4) After all insertions, do a final bounding-box visualization
    rstar.visualize()

    # 5) Now scatter *all* points in black on top
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    plt.scatter(all_x, all_y, c="black", s=15, zorder=5)

    # Final hold
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
