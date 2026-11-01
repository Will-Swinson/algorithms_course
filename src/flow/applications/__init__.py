"""Classic problems reduced to max-flow / min-cost flow: bipartite
matching and assignment, graph-cut image segmentation, and scheduling
/ resource allocation (job-machine, staffing, deadline scheduling,
tournament elimination)."""

from src.flow.applications.matching import (
    MatchingResult,
    assignment_from_matrix,
    bipartite_matching,
    is_matching,
    min_cost_assignment,
)
from src.flow.applications.image_seg import (
    SegmentationResult,
    segment_image,
    segmentation_scores,
    threshold_segment,
)
from src.flow.applications.scheduling import (
    AllocationResult,
    ScheduleResult,
    allocate,
    assign_jobs_to_machines,
    schedule_jobs,
    staff_projects,
    team_eliminated,
    tournament_elimination,
)
