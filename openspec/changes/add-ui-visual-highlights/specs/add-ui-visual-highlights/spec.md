## ADDED Requirements

### Requirement: Same-camera highlight outline
The system SHALL draw a Yellow border around retrieved match images that share the same camera ID as the query image.

#### Scenario: Same camera match
- **WHEN** a gallery match has the same camera ID as the query
- **THEN** it is displayed with an 8-pixel Yellow border and the label "(Same Cam)" appended to its caption.

### Requirement: CA-Jaccard rank improvement outline
The system SHALL draw a Green border around retrieved match images whose rank position improved after applying CA-Jaccard re-ranking.

#### Scenario: Rank improvement match
- **WHEN** CA-Jaccard is enabled and a match's rank is better than its baseline position
- **THEN** it is displayed with an 8-pixel Green border and the label "(Improved from X!)" appended to its caption.
