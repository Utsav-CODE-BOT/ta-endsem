import logging
from typing import Dict, List, Optional
import networkx as nx

logger = logging.getLogger(__name__)


class RelationshipGraphBuilder:
    """
    Constructs multi-relational graphs linking:
    - Issue nodes
    - Candidate cause nodes
    - Aspect nodes
    - Entity / System nodes
    Exports interactive graph JSON schemas for React / WebGL / SVG rendering.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(
        self,
        clusters: List[Dict[str, any]],
        hypotheses: List[Dict[str, any]],
        aspects_summary: Optional[Dict[str, int]] = None,
    ) -> Dict[str, any]:
        """
        Builds a directed graph linking issues, aspects, entities, and candidate causes.
        """
        G = nx.DiGraph()

        # 1. Add Issue nodes
        for cl in clusters:
            issue_id = f"issue_{cl['cluster_id']}"
            G.add_node(
                issue_id,
                id=issue_id,
                label=cl["name"],
                type="issue",
                size=max(20, min(50, int(cl["size"] / 5))),
                group="issues",
                metadata={"size": cl["size"], "percentage": cl["percentage"]},
            )

        # 2. Add Candidate Cause / Factor nodes and edges
        for idx, hyp in enumerate(hypotheses):
            factor_id = f"factor_{hyp['candidate_factor'].lower().replace(' ', '_')}"
            factor_label = hyp["candidate_factor"]
            issue_id = f"issue_{hyp['cluster_id']}"

            if not G.has_node(factor_id):
                G.add_node(
                    factor_id,
                    id=factor_id,
                    label=factor_label,
                    type="candidate_cause",
                    size=24,
                    group="causes",
                    metadata={
                        "confidence": hyp["confidence"],
                        "lift": hyp["lift"],
                        "type": hyp.get("factor_type", "Candidate Cause"),
                    },
                )

            # Edge: Candidate Cause -> Associated With -> Issue
            edge_weight = float(hyp["confidence"])
            G.add_edge(
                factor_id,
                issue_id,
                source=factor_id,
                target=issue_id,
                relationship="associated_with",
                weight=round(edge_weight, 2),
                label=f"Lift {hyp['lift']}x",
            )

        # 3. Add top Aspect nodes if provided
        if aspects_summary:
            for aspect, count in list(aspects_summary.items())[:8]:
                aspect_id = f"aspect_{aspect.lower().replace(' ', '_')}"
                if not G.has_node(aspect_id):
                    G.add_node(
                        aspect_id,
                        id=aspect_id,
                        label=aspect,
                        type="aspect",
                        size=20,
                        group="aspects",
                        metadata={"mention_count": count},
                    )

        # Format output for frontend force-directed or interactive network graph
        nodes = []
        for node_id, data in G.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "type": data.get("type", "node"),
                "size": data.get("size", 20),
                "group": data.get("group", "default"),
                "metadata": data.get("metadata", {}),
            })

        links = []
        for u, v, data in G.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "relationship": data.get("relationship", "linked_to"),
                "weight": data.get("weight", 1.0),
                "label": data.get("label", ""),
            })

        return {
            "nodes": nodes,
            "links": links,
            "total_nodes": len(nodes),
            "total_edges": len(links),
        }
