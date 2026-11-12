
**shortcut_smooth** randomly selects two nonadjacent points on the path and tries to replace the detour between them with a direct edge:
- If `Cs.is_edge_valid(out[i], out[j])` passes, it removes the intermediate points.
- If the direct edge is invalid, it keeps the original path and tries another pair.
- It stops when fewer than three points remain or it has used all its attempts.