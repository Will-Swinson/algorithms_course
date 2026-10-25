"""Approximation algorithms with provable guarantees: greedy (Vertex
Cover 2-approximation, facility-location star greedy), LP relaxation
and rounding (Set Cover, weighted Vertex Cover), metric TSP (MST
doubling, Christofides), and randomized Max-Cut with local search."""

from src.approximation.vertex_cover_2approx import (
    CoverResult,
    lp_rounding_vertex_cover,
    lp_vertex_cover,
    matching_vertex_cover,
)
from src.approximation.set_cover_lpround import (
    SetCoverInstance,
    deterministic_rounding,
    exact_set_cover,
    greedy_bad_instance,
    greedy_set_cover,
    is_cover,
    lp_relaxation,
    random_set_cover,
    randomized_rounding,
)
from src.approximation.facility_location_greedy import (
    FacilityInstance,
    evaluate,
    exact_facility_location,
    greedy_facility_location,
    local_search_facility_location,
    random_metric_instance,
)
from src.approximation.tsp_metric_approx import (
    TourResult,
    christofides_tour,
    mst_doubling_tour,
    optimal_tour,
)
from src.approximation.maxcut_randomized import (
    CutResult,
    cut_value,
    derandomized_cut,
    exact_maxcut,
    expected_random_cut,
    local_search_cut,
    random_cut,
    total_weight,
)
