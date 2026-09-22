"""course_output/range_net_module.py

The built-in range-net "module" -- one section of netting (4 stacked
wire panels + the two RivieraCC fence posts at its ends + the buried
brick anchor). Shipped with the repo, NOT a per-course collection
template (see the user's "built-in like parking" decision).

The template (templates/rangenets.course) is 8 m per module; every
dimension below is scaled up from that by MODEL_TO_REAL_SCALE (31/17 --
the real range's post count vs. the unscaled model's), so MODULE_LENGTH_M
is NOT 8 m -- see MODEL_TO_REAL_SCALE.

The module frame:
  - origin (0, 0) = the FIRST post of the module (local +Z = the fence
    run direction; heading 0 = +Z, same convention as collections.py).
  - the SECOND post sits at local (0, MODULE_LENGTH_M) -- i.e. the
    module's far end. When tiled, the next module's first post IS this
    post (the "string of pearls" seam), so the tiler must NOT re-emit
    a post at the far end of an already-tiled interval.
  - the 4 wire panels are ANCHORED at local (0, 0) -- they all share the
    same anchor point and differ only in Y (dy) -- their 8 m extent
    comes from the prefab model's own long axis (rotated 90 deg so it
    points along +Z), not from a dz offset. Verified against
    templates/rangenets.course (placedObjects3.json): all 4 panels sit
    at z=0 in the template, y = 3.0/7.2/11.4/15.6 m (4.2 m spacing,
    4-high).
  - the brick anchor is also anchored at local (0, 0) (buried ~2.5 m
    below the 1 m ground datum of the template course -- it's a post
    anchor, see the user's note).

dy is the height above the template course's ground datum (1 m, per the
user's "course-wide stamp aligns the blank template at 1 m elevation,
subtract 1 m" note). At write-objects time,
course_output/collections.py:apply_terrain_heights resolves each member
to an absolute y = target terrain height at (x, z) + output_height_shift_m
+ dy, so the net re-grounds itself on whatever terrain it lands on --
no heightmap needed in the generate step.

Source: templates/rangenets.course (added to the repo by the user),
extracted to a working folder with util/course_extract.py and read off
CourseDescription_nodes/placedObjects3.json.
"""

from __future__ import annotations

# The real range has 17 posts along its run; the built-in module (at its
# original template scale) tiles out to 31 over the same OSM way -- the
# module is undersized. Scale every dimension (length, prefab scale,
# height) up by this ratio so post spacing -- and everything else -- is
# proportionately bigger, closer to the real 17-post spacing.
MODEL_TO_REAL_SCALE = 31 / 17

# Module length along its forward axis (local +Z) -- one fence segment
# (4 wire panels stacked 4-high), scaled up from the template's 8 m.
MODULE_LENGTH_M = 8.0 * MODEL_TO_REAL_SCALE

# In-game, the whole module was landing too high relative to the
# ground -- subtracted from every member's dy below, on top of the 1 m
# ground datum. Tuned at the template's original 8 m scale to 2.4 m; it's
# a fixed point in the module's own local frame (where its "too high"
# vertical geometry sits relative to its origin), so it scales up with
# everything else, same as MODULE_LENGTH_M.
GROUND_DROP_M = 2.4 * MODEL_TO_REAL_SCALE

# The two RivieraCC fence posts -- one at each end of the module.
# rotation_deg is RELATIVE TO THE MODULE HEADING (0 = +Z). The template
# has them at rotation 0 and 90; the 90 on the far post is the
# template author's choice (the post is visually symmetric in yaw so
# either reads the same).
POST_PATH = "Assets/CourseGen/Detail/Walls/RivieraCC_Fence01AAPostAPrefab"
POST_SCALE = 5.0 * MODEL_TO_REAL_SCALE
# y=6.0 in the template, minus the 1 m ground datum, scaled up, minus
# the (unscaled) ground-alignment drop.
POST_DY = 5.0 * MODEL_TO_REAL_SCALE - GROUND_DROP_M

POSTS = [
    {"path": POST_PATH, "dz": 0.0,  "rotation_deg": 0.0, "scale": POST_SCALE, "dy": POST_DY},
    {"path": POST_PATH, "dz": MODULE_LENGTH_M, "rotation_deg": 90.0, "scale": POST_SCALE, "dy": POST_DY},
]

# The 4 stacked wire panels -- all anchored at local (0, 0), spanning
# the module's length via the prefab model's long axis (rotated 90 deg).
WIRE_PATH = "Assets/CourseGen/Detail/Walls/WireFenceACPrefab"
WIRE_SCALE = 2.0 * MODEL_TO_REAL_SCALE
WIRE_ROTATION_DEG = 90.0  # the template's 90 deg yaw (model's long axis -> +Z)

# (dy, label) for the 4 stacked panels -- y = 3.0/7.2/11.4/15.6 in the
# template, minus the 1 m ground datum, scaled up, minus the ground
# drop. 4.2 m template spacing, scaled the same as everything else.
WIRE_PANELS = [
    (2.0  * MODEL_TO_REAL_SCALE - GROUND_DROP_M, "L1"),
    (6.2  * MODEL_TO_REAL_SCALE - GROUND_DROP_M, "L2"),
    (10.4 * MODEL_TO_REAL_SCALE - GROUND_DROP_M, "L3"),
    (14.6 * MODEL_TO_REAL_SCALE - GROUND_DROP_M, "L4"),
]

# The buried brick anchor at the module's start (a post anchor --
# included per the user's note). dy = -1.5 - 1.0 = -2.5 (2.5 m below
# the 1 m ground datum) in the template, scaled up, minus the ground
# drop.
BRICK_PATH = "Assets/CourseGen/Detail/Walls/BrickWallsHighRailsAPostAPrefab"
BRICK_SCALE = 3.0 * MODEL_TO_REAL_SCALE
BRICK_ROTATION_DEG = 0.0
BRICK_DY = -2.5 * MODEL_TO_REAL_SCALE - GROUND_DROP_M


def post_anchor() -> dict:
    """The buried brick/concrete post anchor, on its own -- one belongs
    at every post, not just an interval's start. span_members() places
    one per interval (anchored alongside that interval's start post);
    the tiler places one more of these at a chain's terminal post (the
    end of the last interval), which span_members() never reaches since
    it isn't the start of any interval."""
    return {"path": BRICK_PATH, "dz": 0.0, "rotation_deg": BRICK_ROTATION_DEG,
            "scale": BRICK_SCALE, "dy": BRICK_DY}


def span_members() -> list[dict]:
    """The per-interval members: the 4 stacked wire panels + the brick
    anchor. All anchored at local (0, 0); the wire panels' 8 m extent
    comes from the prefab model's long axis (WIRE_ROTATION_DEG)."""
    out: list[dict] = [
        {"path": WIRE_PATH, "dz": 0.0, "rotation_deg": WIRE_ROTATION_DEG,
         "scale": WIRE_SCALE, "dy": dy}
        for dy, _label in WIRE_PANELS
    ]
    out.append(post_anchor())
    return out


def start_post() -> dict:
    """The module's FIRST post (local (0, 0)) -- one per 8 m node, shared
    at the seams. The tiler emits this at every node (0, 8, 16, ...)
    and the span_members() at every interval start."""
    return dict(POSTS[0])


def far_post() -> dict:
    """The module's SECOND post (local (0, MODULE_LENGTH_M)) -- the far
    end of the interval. Only emitted for a SINGLE (non-tiled) range-net
    way; the tiler's shared-seam logic means a tiled run of N intervals
    emits N+1 start posts (one per node) and never a far post."""
    return dict(POSTS[1])
