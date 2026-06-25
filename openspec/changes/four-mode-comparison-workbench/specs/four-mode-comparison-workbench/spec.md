## ADDED Requirements

### Requirement: Four-mode evaluation output
The system SHALL process raw and Kalman query crops and produce four top-5 galleries representing the combinations of Kalman tracking and CA-Jaccard re-ranking.

#### Scenario: Running comparison
- **WHEN** the user clicks "Run 4-Mode Comparison"
- **THEN** it generates results for:
  - Mode 1: Kalman OFF | CA-Jaccard OFF
  - Mode 2: Kalman ON | CA-Jaccard OFF
  - Mode 3: Kalman OFF | CA-Jaccard ON
  - Mode 4: Kalman ON | CA-Jaccard ON
  and renders them concurrently.
