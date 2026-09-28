"""One module per ingestion stage: graph, mst, describe, embed.

Each builds a structure or generates text and returns it; persistence belongs to
`data.repositories`. Import the stage module you need directly -- these are
deliberately not re-exported here, so a stage's dependencies stay visible at the
import site.
"""
