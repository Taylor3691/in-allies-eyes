# integrate-localized-caj-reranking Specification

## Purpose
Specification for localized CA-Jaccard re-ranking and camera masking requirements.

## Requirements

### Requirement: Localized CA-Jaccard optimization pool
The system SHALL restrict the Jaccard distance matrix re-ranking calculation to a subset of the top-200 initial matches from the Cosine similarity search.

#### Scenario: Sub-second ranking optimization
- **WHEN** gallery search is executed with CA-Jaccard enabled
- **THEN** re-ranking takes under 0.05 seconds to run.

### Requirement: Camera-aware Jaccard mask mapping
The system SHALL map the query camera ID and the top-200 gallery camera IDs into the `cids` parameter of the re-ranking function.

#### Scenario: Same-camera bias correction
- **WHEN** the camera IDs are supplied to the Jaccard calculation
- **THEN** camera-specific distance offsets are added to correct same-camera matching errors.
