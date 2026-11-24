"""
objective1_citational_graph.py – Objective 1: Citational Knowledge Graph (GraphRAG) Construction & Structural Analysis

Constructs a Citational Knowledge Graph (CKG) from research paper metadata,
computes structural network metrics (PageRank, In-Degree, Betweenness, Louvain Communities),
and provides a GraphRAG context retrieval interface for LLM augmentation.
"""

import os
import re
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Configuration
DATASET_PATH = "final_arxiv_dataset.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class CitationalKnowledgeGraph:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.graph = nx.DiGraph()
        self.df = None
        self.metrics_df = None
        self.communities = []
        
    def load_data(self):
        """Loads research paper dataset and cleans text."""
        print("[1/5] Ingesting research paper dataset...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path).dropna().reset_index(drop=True)
        # Format paper IDs as P_0001, P_0002, etc.
        self.df["paper_id"] = [f"P_{i+1:04d}" for i in range(len(self.df))]
        
        # Synthetic author generation for structured citation network
        sample_authors = [
            "A. Vaswani", "N. Shazeer", "N. Parmar", "J. Uszkoreit", "L. Jones", 
            "A. N. Gomez", "L. Kaiser", "I. Polosukhin", "Y. LeCun", "Y. Bengio",
            "G. Hinton", "A. Radfoad", "I. Sutskever", "D. Amodei", "P. Liang"
        ]
        
        # Assign authors and publication years deterministically
        authors_list = []
        years_list = []
        for idx, row in self.df.iterrows():
            seed = int(re.sub(r"\D", "", row["paper_id"]))
            rnd = random.Random(seed)
            paper_authors = rnd.sample(sample_authors, rnd.randint(1, 3))
            paper_year = rnd.randint(2018, 2024)
            authors_list.append(paper_authors)
            years_list.append(paper_year)
            
        self.df["authors"] = authors_list
        self.df["year"] = years_list
        print(f"   Loaded {len(self.df)} papers with extracted metadata.")
        return self.df

    def build_graph(self, max_papers=2500):
        """Converts research paper metadata into Nodes and Edges."""
        paper_count = len(self.df) if (max_papers is None or max_papers >= len(self.df)) else max_papers
        print(f"[2/5] Constructing Citational Knowledge Graph (using {paper_count} papers)...")
        sub_df = self.df.head(paper_count)
        self.graph.clear()
        
        # 1. Add Nodes
        for _, row in sub_df.iterrows():
            pid = row["paper_id"]
            title = row["title"]
            category = row["category"]
            year = row["year"]
            
            # Paper Node
            self.graph.add_node(
                pid, 
                type="Paper", 
                title=title, 
                category=category, 
                year=year,
                label=f"{pid}: {title[:30]}..."
            )
            
            # Category / Topic Node & HAS_TOPIC edge
            topic_node = f"Topic_{category}"
            self.graph.add_node(topic_node, type="Topic", category=category, label=category)
            self.graph.add_edge(pid, topic_node, relation="HAS_TOPIC")
            
            # Author Nodes & AUTHORED edge
            for author in row["authors"]:
                author_node = f"Author_{author}"
                self.graph.add_node(author_node, type="Author", name=author, label=author)
                self.graph.add_edge(author_node, pid, relation="AUTHORED")
                
        # 2. Build Citation Edges based on shared topics and publication year order
        papers_list = sub_df.to_dict("records")
        for i, p1 in enumerate(papers_list):
            for j, p2 in enumerate(papers_list):
                if i != j and p1["year"] > p2["year"]:
                    # Higher probability of citation if categories match
                    prob = 0.15 if p1["category"] == p2["category"] else 0.015
                    seed = hash(p1["paper_id"] + p2["paper_id"]) % 10000
                    rnd = random.Random(seed)
                    if rnd.random() < prob:
                        # Directed edge: P1 CITES P2
                        self.graph.add_edge(p1["paper_id"], p2["paper_id"], relation="CITES")
                        
        # Log Graph Validation Statistics
        paper_nodes_count = len([n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"])
        author_nodes_count = len([n for n, d in self.graph.nodes(data=True) if d.get("type") == "Author"])
        category_nodes_count = len([n for n, d in self.graph.nodes(data=True) if d.get("type") == "Topic"])
        
        print(f"   Graph loaded successfully:")
        print(f"     Nodes: {self.graph.number_of_nodes()}")
        print(f"     Edges: {self.graph.number_of_edges()}")
        print(f"     Paper nodes: {paper_nodes_count}")
        print(f"     Author nodes: {author_nodes_count}")
        print(f"     Category nodes: {category_nodes_count}")
        
        return self.graph

    def compute_structural_analysis(self):
        """Computes structural graph metrics: PageRank, Centralities, Community Detection."""
        print("[3/5] Computing structural graph analysis (Centrality & PageRank)...")
        
        # Isolate Paper-only Subgraph for citation topology
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"]
        
        if not paper_nodes or self.graph.number_of_nodes() == 0:
            print("   Warning: Graph contains zero paper nodes. Returning empty metrics dataframe.")
            self.metrics_df = pd.DataFrame(columns=[
                "Paper_ID", "Title", "Category", "Year", 
                "In_Degree_Citations", "Out_Degree_References", 
                "PageRank", "Betweenness_Centrality", "Research_Community"
            ])
            return self.metrics_df

        paper_subgraph = self.graph.subgraph(paper_nodes)
        
        # Centrality & Ranking Metrics
        in_degree = dict(paper_subgraph.in_degree())
        out_degree = dict(paper_subgraph.out_degree())
        
        # Safe PageRank computation
        try:
            pagerank = nx.pagerank(paper_subgraph, alpha=0.85) if len(paper_subgraph) > 0 else {}
        except Exception as e:
            print(f"   PageRank computation warning: {e}")
            pagerank = {n: 1.0 / len(paper_subgraph) for n in paper_subgraph.nodes()}
            
        # Safe Betweenness Centrality
        try:
            betweenness = nx.betweenness_centrality(paper_subgraph) if len(paper_subgraph) > 0 else {}
        except Exception as e:
            print(f"   Betweenness computation warning: {e}")
            betweenness = {n: 0.0 for n in paper_subgraph.nodes()}
        
        # Community Detection (Louvain algorithm on undirected citation projection)
        undirected_proj = paper_subgraph.to_undirected()
        try:
            self.communities = list(nx.community.louvain_communities(undirected_proj, random_state=42))
        except Exception:
            self.communities = [set(paper_nodes)]
            
        community_map = {}
        for comm_id, comm_nodes in enumerate(self.communities, 1):
            for node in comm_nodes:
                community_map[node] = comm_id
                
        metrics = []
        for pid in paper_nodes:
            node_data = self.graph.nodes[pid]
            metrics.append({
                "Paper_ID": pid,
                "Title": node_data.get("title", ""),
                "Category": node_data.get("category", ""),
                "Year": node_data.get("year", 2022),
                "In_Degree_Citations": in_degree.get(pid, 0),
                "Out_Degree_References": out_degree.get(pid, 0),
                "PageRank": round(pagerank.get(pid, 0.0), 6),
                "Betweenness_Centrality": round(betweenness.get(pid, 0.0), 6),
                "Research_Community": community_map.get(pid, 1)
            })
            
        self.metrics_df = pd.DataFrame(metrics).sort_values(by="PageRank", ascending=False)
        metrics_csv_path = os.path.join(RESULTS_DIR, "obj1_citational_graph_metrics.csv")
        self.metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"   Structural metrics calculated and saved to -> {metrics_csv_path}")
        return self.metrics_df

    def visualize_graph(self, max_display_nodes=60):
        """Generates a visual network topology plot."""
        print("[4/5] Generating graph visualization plot...")
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"][:max_display_nodes]
        
        chart_path = os.path.join(RESULTS_DIR, "obj1_knowledge_graph.png")
        if not paper_nodes or self.graph.number_of_nodes() == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Empty Graph Data", ha="center", va="center")
            plt.savefig(chart_path)
            plt.close()
            return chart_path

        sub = self.graph.subgraph(paper_nodes)
        
        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(sub, k=0.4, seed=42)
        
        # Draw nodes with size proportional to PageRank
        try:
            pr = nx.pagerank(sub, alpha=0.85) if len(sub) > 0 else {n: 0.01 for n in sub.nodes()}
        except Exception:
            pr = {n: 0.01 for n in sub.nodes()}
            
        node_sizes = [max(150, pr[n] * 15000) for n in sub.nodes()]
        
        nx.draw_networkx_nodes(sub, pos, node_size=node_sizes, node_color="#4CAF50", alpha=0.85)
        nx.draw_networkx_edges(sub, pos, arrowstyle="->", arrowsize=12, edge_color="#90A4AE", alpha=0.6)
        
        labels = {n: n for n in sub.nodes()}
        nx.draw_networkx_labels(sub, pos, labels=labels, font_size=8, font_color="#1A237E", font_weight="bold")
        
        plt.title("Citational Knowledge Graph Topology (Node size proportional to PageRank)", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   Visualization chart saved -> {chart_path}")
        return chart_path

    def generate_interactive_html(self, center_query=None, category_filter="All", max_nodes=100, show_authors=True, show_topics=True):
        """
        Generates an interactive, dynamic Vis.js Network HTML representation of the Knowledge Graph.
        Supports drag, zoom, hover tooltips, click selection, and filtering by category or centered paper.
        """
        import json
        
        # 1. Determine target paper node if center_query is provided
        target_pid = None
        if center_query and str(center_query).strip():
            target_pid, _ = self.normalize_paper_id(center_query)
            
        # 2. Select paper nodes to include
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"]
        
        if category_filter and category_filter != "All":
            paper_nodes = [n for n in paper_nodes if self.graph.nodes[n].get("category") == category_filter]
            
        if target_pid and target_pid in self.graph:
            # Include 1-hop and 2-hop neighborhood of target paper
            neighborhood = {target_pid}
            for neighbor in self.graph.successors(target_pid):
                if self.graph.nodes[neighbor].get("type") == "Paper":
                    neighborhood.add(neighbor)
            for pred in self.graph.predecessors(target_pid):
                if self.graph.nodes[pred].get("type") == "Paper":
                    neighborhood.add(pred)
                    
            # Prioritize neighborhood nodes, then pad with top PageRank nodes up to max_nodes
            sorted_papers = [n for n in paper_nodes if n in neighborhood]
            other_papers = [n for n in paper_nodes if n not in neighborhood]
            paper_nodes = (sorted_papers + other_papers)[:max_nodes]
        else:
            # Sort paper_nodes by PageRank if metrics_df is available
            if self.metrics_df is not None and not self.metrics_df.empty:
                pr_map = dict(zip(self.metrics_df["Paper_ID"], self.metrics_df["PageRank"]))
                paper_nodes = sorted(paper_nodes, key=lambda n: pr_map.get(n, 0), reverse=True)
            paper_nodes = paper_nodes[:max_nodes]
            
        selected_nodes_set = set(paper_nodes)
        
        # Include associated authors and topics if enabled
        if show_authors:
            for pid in paper_nodes:
                for pred in self.graph.predecessors(pid):
                    if self.graph.nodes[pred].get("type") == "Author":
                        selected_nodes_set.add(pred)
                        
        if show_topics:
            for pid in paper_nodes:
                for succ in self.graph.successors(pid):
                    if self.graph.nodes[succ].get("type") == "Topic":
                        selected_nodes_set.add(succ)

        # Build Vis.js Nodes, Edges & Metadata
        vis_nodes = []
        vis_edges = []
        node_metadata = {}
        
        # Helper for PageRank & Degree lookups
        pr_dict = {}
        in_degree_dict = {}
        out_degree_dict = {}
        if self.metrics_df is not None and not self.metrics_df.empty:
            pr_dict = dict(zip(self.metrics_df["Paper_ID"], self.metrics_df["PageRank"]))
            in_degree_dict = dict(zip(self.metrics_df["Paper_ID"], self.metrics_df["In_Degree_Citations"]))
            out_degree_dict = dict(zip(self.metrics_df["Paper_ID"], self.metrics_df["Out_Degree_References"]))
            
        for n in selected_nodes_set:
            d = self.graph.nodes[n]
            node_type = d.get("type", "Paper")
            
            if node_type == "Paper":
                title = d.get("title", n)
                cat = d.get("category", "N/A")
                year = d.get("year", "N/A")
                pr_val = pr_dict.get(n, 0.001)
                in_deg = in_degree_dict.get(n, 0)
                out_deg = out_degree_dict.get(n, 0)
                
                # Highlight target node
                is_target = (n == target_pid)
                size = 35 if is_target else max(15, min(45, int(pr_val * 15000)))
                color = "#FFD700" if is_target else "#10B981" # Gold for target, Emerald green for papers
                shape = "star" if is_target else "dot"
                
                tooltip = (
                    f"<b>Paper ID:</b> {n}<br>"
                    f"<b>Title:</b> {title}<br>"
                    f"<b>Category:</b> {cat}<br>"
                    f"<b>Year:</b> {year}<br>"
                    f"<b>PageRank Prestige:</b> {pr_val:.6f}<br>"
                    f"<b>Citations Received:</b> {in_deg}<br>"
                    f"<b>Click node to highlight neighborhood</b>"
                )
                
                vis_nodes.append({
                    "id": n,
                    "label": f"{n}\n{title[:20]}...",
                    "title": tooltip,
                    "shape": shape,
                    "size": size,
                    "color": {"background": color, "border": "#065F46"},
                    "font": {"color": "#FFFFFF", "size": 12}
                })
                
                node_metadata[n] = {
                    "type": "Paper",
                    "title": title,
                    "category": cat,
                    "year": year,
                    "pagerank": pr_val,
                    "in_degree": in_deg,
                    "out_degree": out_deg
                }
                
            elif node_type == "Author":
                name = d.get("name", n)
                tooltip = f"<b>Author Node:</b> {name}<br>Click to highlight authored papers"
                vis_nodes.append({
                    "id": n,
                    "label": name,
                    "title": tooltip,
                    "shape": "diamond",
                    "size": 18,
                    "color": {"background": "#F59E0B", "border": "#B45309"},
                    "font": {"color": "#FEF3C7", "size": 11}
                })
                node_metadata[n] = {"type": "Author", "name": name, "title": name}
                
            elif node_type == "Topic":
                cat_name = d.get("category", n)
                tooltip = f"<b>Category Node:</b> {cat_name}<br>Click to highlight category cluster"
                vis_nodes.append({
                    "id": n,
                    "label": f"[{cat_name}]",
                    "title": tooltip,
                    "shape": "hexagon",
                    "size": 22,
                    "color": {"background": "#8B5CF6", "border": "#5B21B6"},
                    "font": {"color": "#EDE9FE", "size": 12}
                })
                node_metadata[n] = {"type": "Category", "category": cat_name, "title": cat_name}

        # Add edges between selected nodes
        subgraph = self.graph.subgraph(selected_nodes_set)
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", "")
            if rel == "CITES":
                vis_edges.append({
                    "from": u,
                    "to": v,
                    "arrows": "to",
                    "label": "CITES",
                    "color": {"color": "#64748B", "highlight": "#38BDF8"},
                    "width": 1.5
                })
            elif rel == "AUTHORED":
                vis_edges.append({
                    "from": u,
                    "to": v,
                    "arrows": "to",
                    "label": "AUTHORED",
                    "color": {"color": "#F59E0B", "highlight": "#FBBF24"},
                    "dashes": True,
                    "width": 1.0
                })
            elif rel == "HAS_TOPIC":
                vis_edges.append({
                    "from": u,
                    "to": v,
                    "arrows": "to",
                    "label": "HAS_TOPIC",
                    "color": {"color": "#A855F7", "highlight": "#C084FC"},
                    "dashes": True,
                    "width": 1.0
                })

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)
        metadata_json = json.dumps(node_metadata)
        
        # Ultra-Dynamic Vis.js Network Canvas with Live Layout Switcher, HD PNG Export, Double-Click Focus & HUD Overlay
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
          <style type="text/css">
            body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background-color: #0F172A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            #mynetwork {{ width: 100%; height: 680px; background-color: #0F172A; }}
            
            .toolbar {{
              position: absolute;
              top: 12px;
              left: 12px;
              z-index: 100;
              display: flex;
              gap: 8px;
              flex-wrap: wrap;
              background: rgba(15, 23, 42, 0.92);
              padding: 8px 14px;
              border-radius: 10px;
              border: 1px solid #334155;
              box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            }}
            .toolbar input, .toolbar select {{
              background: #1E293B;
              border: 1px solid #475569;
              color: white;
              padding: 6px 10px;
              border-radius: 6px;
              font-size: 12px;
              outline: none;
            }}
            .toolbar button {{
              background: #0EA5E9;
              color: white;
              border: none;
              padding: 6px 12px;
              border-radius: 6px;
              cursor: pointer;
              font-size: 12px;
              font-weight: 600;
              transition: all 0.2s ease;
            }}
            .toolbar button:hover {{ background: #0284C7; transform: translateY(-1px); }}
            
            .hud-overlay {{
              position: absolute;
              bottom: 16px;
              right: 16px;
              background: rgba(15, 23, 42, 0.9);
              border: 1px solid #334155;
              border-radius: 8px;
              padding: 10px 14px;
              color: #94A3B8;
              font-size: 11px;
              z-index: 99;
              box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .hud-stat {{ display: flex; justify-content: space-between; gap: 15px; margin-bottom: 3px; }}
            .hud-stat span {{ color: #38BDF8; font-weight: bold; }}
            
            .legend {{
              position: absolute;
              top: 12px;
              right: 12px;
              background: rgba(15, 23, 42, 0.92);
              padding: 10px 14px;
              border-radius: 10px;
              color: white;
              border: 1px solid #334155;
              font-size: 12px;
              z-index: 99;
            }}
            .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
            .legend-color {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
            
            #detailsPanel {{
              position: absolute;
              bottom: 16px;
              left: 16px;
              width: 360px;
              max-height: 280px;
              background: rgba(15, 23, 42, 0.96);
              border: 1px solid #38BDF8;
              border-radius: 12px;
              color: white;
              padding: 16px;
              box-shadow: 0 12px 30px -5px rgba(0,0,0,0.6);
              font-size: 12px;
              display: none;
              overflow-y: auto;
              z-index: 101;
            }}
            #detailsPanel h4 {{ margin: 0 0 8px 0; color: #38BDF8; font-size: 14px; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
            .close-btn {{ position: absolute; top: 10px; right: 14px; cursor: pointer; color: #94A3B8; font-weight: bold; font-size: 14px; }}
          </style>
        </head>
        <body>
          <div class="toolbar">
            <input type="text" id="canvasSearch" placeholder="🔍 Search canvas nodes..." onkeyup="filterCanvasNodes()">
            <select id="layoutSelect" onchange="changeLayout(this.value)">
              <option value="force">🌀 Organic Force-Directed</option>
              <option value="tree">🌲 Hierarchical Citation Flow</option>
            </select>
            <button onclick="zoomFit()">🔍 Fit View</button>
            <button onclick="togglePhysics()" id="physicsBtn">⏸️ Freeze Physics</button>
            <button onclick="resetHighlight()">🔄 Reset View</button>
            <button onclick="exportPNG()" style="background:#10B981;">📸 Export HD PNG</button>
          </div>
          
          <div class="legend">
            <div style="font-weight: bold; margin-bottom: 6px; color: #38BDF8;">Graph Legend</div>
            <div class="legend-item"><span class="legend-color" style="background:#FFD700;"></span> Searched Target</div>
            <div class="legend-item"><span class="legend-color" style="background:#10B981;"></span> Paper Node (PageRank Size)</div>
            <div class="legend-item"><span class="legend-color" style="background:#F59E0B;"></span> Author Node</div>
            <div class="legend-item"><span class="legend-color" style="background:#8B5CF6;"></span> Category Node</div>
          </div>

          <div class="hud-overlay">
            <div class="hud-stat">Visible Nodes: <span id="hudNodes">{len(vis_nodes)}</span></div>
            <div class="hud-stat">Visible Edges: <span id="hudEdges">{len(vis_edges)}</span></div>
            <div class="hud-stat">Physics Engine: <span id="hudPhysics" style="color:#10B981;">Active</span></div>
          </div>

          <div id="detailsPanel">
            <span class="close-btn" onclick="closeDetails()">✕</span>
            <h4 id="panelTitle">Paper Details</h4>
            <div id="panelContent">Click any node to inspect details...</div>
          </div>

          <div id="mynetwork"></div>

          <script type="text/javascript">
            var rawNodes = {nodes_json};
            var rawEdges = {edges_json};
            var nodeMetadata = {metadata_json};

            var nodes = new vis.DataSet(rawNodes);
            var edges = new vis.DataSet(rawEdges);
            var container = document.getElementById('mynetwork');
            var data = {{ nodes: nodes, edges: edges }};

            var physicsEnabled = true;
            var forceOptions = {{
              nodes: {{ font: {{ color: '#FFFFFF', face: 'Segoe UI' }}, borderWidth: 2 }},
              edges: {{ font: {{ color: '#94A3B8', size: 10, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
              interaction: {{ hover: true, tooltipDelay: 50, navigationButtons: true, zoomView: true, dragView: true }},
              layout: {{ hierarchical: false }},
              physics: {{
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }},
                stabilization: {{ iterations: 150 }}
              }}
            }};

            var network = new vis.Network(container, data, forceOptions);

            // Single Click Handler: Highlight neighborhood & show side panel
            network.on("click", function(params) {{
              if (params.nodes.length > 0) {{
                var selectedNode = params.nodes[0];
                highlightNeighbors(selectedNode);
                showNodeDetails(selectedNode);
              }} else {{
                resetHighlight();
                closeDetails();
              }}
            }});

            // Double Click Handler: Smooth Camera Zoom & Focus Animation
            network.on("doubleClick", function(params) {{
              if (params.nodes.length > 0) {{
                var selectedNode = params.nodes[0];
                network.focus(selectedNode, {{
                  scale: 1.3,
                  animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }}
                }});
              }}
            }});

            function highlightNeighbors(selectedNode) {{
              var connected = network.getConnectedNodes(selectedNode);
              connected.push(selectedNode);

              var updateArray = [];
              rawNodes.forEach(function(n) {{
                if (connected.includes(n.id)) {{
                  updateArray.push({{ id: n.id, hidden: false, opacity: 1.0 }});
                }} else {{
                  updateArray.push({{ id: n.id, hidden: false, opacity: 0.15 }});
                }}
              }});
              nodes.update(updateArray);
            }}

            function resetHighlight() {{
              var updateArray = rawNodes.map(function(n) {{
                return {{ id: n.id, hidden: false, opacity: 1.0 }};
              }});
              nodes.update(updateArray);
            }}

            function filterCanvasNodes() {{
              var query = document.getElementById('canvasSearch').value.toLowerCase().trim();
              if (!query) {{ resetHighlight(); return; }}
              var updateArray = [];
              rawNodes.forEach(function(n) {{
                var label = (n.label || '').toLowerCase();
                var nodeTitle = (n.title || '').toLowerCase();
                if (label.includes(query) || nodeTitle.includes(query)) {{
                  updateArray.push({{ id: n.id, opacity: 1.0 }});
                }} else {{
                  updateArray.push({{ id: n.id, opacity: 0.15 }});
                }}
              }});
              nodes.update(updateArray);
            }}

            function showNodeDetails(nodeId) {{
              var meta = nodeMetadata[nodeId] || {{}};
              var panel = document.getElementById('detailsPanel');
              var titleElem = document.getElementById('panelTitle');
              var contentElem = document.getElementById('panelContent');

              titleElem.innerText = (meta.type || 'Node') + ": " + nodeId;
              if (meta.type === 'Paper') {{
                contentElem.innerHTML = 
                  "<b>Title:</b> " + (meta.title || nodeId) + "<br>" +
                  "<b>Category:</b> " + (meta.category || 'N/A') + "<br>" +
                  "<b>Year:</b> " + (meta.year || 'N/A') + "<br>" +
                  "<b>PageRank Prestige:</b> " + (meta.pagerank ? meta.pagerank.toFixed(6) : 'N/A') + "<br>" +
                  "<b>Direct Citations:</b> " + (meta.in_degree || 0) + " received, " + (meta.out_degree || 0) + " references";
              }} else if (meta.type === 'Author') {{
                contentElem.innerHTML = "<b>Author Name:</b> " + (meta.name || nodeId);
              }} else {{
                contentElem.innerHTML = "<b>Category Domain:</b> " + (meta.category || nodeId);
              }}
              
              panel.style.display = 'block';
            }}

            function closeDetails() {{
              document.getElementById('detailsPanel').style.display = 'none';
            }}

            function zoomFit() {{ network.fit({{ animation: {{ duration: 600 }} }}); }}

            function togglePhysics() {{
              physicsEnabled = !physicsEnabled;
              network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
              document.getElementById('physicsBtn').innerText = physicsEnabled ? '⏸️ Freeze Physics' : '▶️ Unfreeze Physics';
              document.getElementById('hudPhysics').innerText = physicsEnabled ? 'Active' : 'Paused';
              document.getElementById('hudPhysics').style.color = physicsEnabled ? '#10B981' : '#EF4444';
            }}

            function changeLayout(mode) {{
              if (mode === 'tree') {{
                network.setOptions({{
                  layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed', levelSeparation: 150, nodeSpacing: 100 }} }},
                  physics: {{ hierarchicalRepulsion: {{ nodeDistance: 120 }} }}
                }});
              }} else {{
                network.setOptions({{
                  layout: {{ hierarchical: false }},
                  physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }}
                  }}
                }});
              }}
            }}

            function exportPNG() {{
              var canvas = container.getElementsByTagName('canvas')[0];
              if (canvas) {{
                var image = canvas.toDataURL('image/png');
                var link = document.createElement('a');
                link.download = 'Knowledge_Graph_Screenshot.png';
                link.href = image;
                link.click();
              }}
            }}
          </script>
        </body>
        </html>
        """
        
        # Return iframe html container to embed seamlessly inside Gradio HTML component
        escaped_html = html_code.replace('"', '&quot;')
        iframe_wrapper = f"""
        <iframe srcdoc="{escaped_html}" style="width: 100%; height: 700px; border: none; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);"></iframe>
        """
        return iframe_wrapper

    def normalize_paper_id(self, query):
        """
        Flexible query resolution supporting:
        - Exact ID: 'P_0001'
        - Short ID / different padding: 'P_001', 'P001', 'P_1', '1'
        - Case insensitive ID: 'p_0001', 'p_001'
        - Title substring search: 'Deep Learning', 'Attention'
        """
        query_str = str(query).strip()
        if not query_str:
            return None, "Empty query string provided."

        # 1. Direct node match
        if query_str in self.graph and self.graph.nodes[query_str].get("type") == "Paper":
            return query_str, None

        # 2. Case-insensitive node match
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "Paper" and n.lower() == query_str.lower():
                return n, None

        # 3. Numeric extraction matching P_XXXX format
        digits = re.findall(r"\d+", query_str)
        if digits:
            num = int(digits[0])
            formatted_id = f"P_{num:04d}"
            if formatted_id in self.graph:
                return formatted_id, f"Normalized query '{query_str}' to Paper ID '{formatted_id}'."

        # 4. Title substring match (case-insensitive)
        matching_nodes = []
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "Paper" and query_str.lower() in d.get("title", "").lower():
                matching_nodes.append((n, d.get("title")))
                
        if matching_nodes:
            matched_id, matched_title = matching_nodes[0]
            return matched_id, f"Found paper by title substring match: '{matched_title}' ({matched_id})."

        return None, f"Paper ID or title query '{query_str}' was not found in the dataset."

    def retrieve_graphrag_context(self, query_text_or_id):
        """
        GraphRAG Retrieval Engine: Retrieves 2-hop structural subgraphs for target paper or query term,
        and constructs a structured GraphRAG system prompt.
        """
        target_pid, note = self.normalize_paper_id(query_text_or_id)
        
        fallback_used = False
        if not target_pid:
            fallback_used = True
            # Fallback to highest PageRank paper if available
            if self.metrics_df is not None and not self.metrics_df.empty:
                target_pid = self.metrics_df.iloc[0]["Paper_ID"]
            else:
                paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Paper"]
                target_pid = paper_nodes[0] if paper_nodes else "P_0001"

        paper_data = self.graph.nodes.get(target_pid, {})
        title = paper_data.get("title", "Unknown Title")
        category = paper_data.get("category", "Unknown Category")
        year = paper_data.get("year", 2022)

        # 1-hop and 2-hop neighbors
        cited_papers = []      # Papers cited BY target_pid
        cited_by_papers = []   # Papers CITING target_pid
        authors = []
        topics = []
        graph_relationships = []

        # Outgoing edges (CITES, HAS_TOPIC)
        for neighbor in self.graph.successors(target_pid):
            edge_data = self.graph.edges[target_pid, neighbor]
            rel = edge_data.get("relation", "")
            if rel == "CITES":
                neighbor_title = self.graph.nodes[neighbor].get("title", neighbor)
                cited_papers.append(f"{neighbor_title} ({neighbor})")
                graph_relationships.append(f"({target_pid}) -[:CITES]-> ({neighbor})")
            elif rel == "HAS_TOPIC":
                cat_name = self.graph.nodes[neighbor].get("category", neighbor)
                topics.append(cat_name)
                graph_relationships.append(f"({target_pid}) -[:HAS_TOPIC]-> ({neighbor})")

        # Incoming edges (CITED_BY, AUTHORED)
        for predecessor in self.graph.predecessors(target_pid):
            edge_data = self.graph.edges[predecessor, target_pid]
            rel = edge_data.get("relation", "")
            if rel == "CITES":
                pred_title = self.graph.nodes[predecessor].get("title", predecessor)
                cited_by_papers.append(f"{pred_title} ({predecessor})")
                graph_relationships.append(f"({predecessor}) -[:CITES]-> ({target_pid})")
            elif rel == "AUTHORED":
                author_name = self.graph.nodes[predecessor].get("name", predecessor)
                authors.append(author_name)
                graph_relationships.append(f"({author_name}) -[:AUTHORED]-> ({target_pid})")

        # Rank metrics info
        pr_score = 0.0
        comm_id = 1
        if self.metrics_df is not None and not self.metrics_df.empty:
            match = self.metrics_df[self.metrics_df["Paper_ID"] == target_pid]
            if not match.empty:
                pr_score = match.iloc[0]["PageRank"]
                comm_id = match.iloc[0]["Research_Community"]

        # Related papers from same research community
        related_community_papers = []
        if self.metrics_df is not None and not self.metrics_df.empty:
            comm_matches = self.metrics_df[
                (self.metrics_df["Research_Community"] == comm_id) & 
                (self.metrics_df["Paper_ID"] != target_pid)
            ].head(3)
            for _, r in comm_matches.iterrows():
                related_community_papers.append(f"{r['Title']} ({r['Paper_ID']})")

        # Format header note if query required normalization or fallback
        header_note = ""
        if fallback_used:
            header_note = f"[GraphRAG Note: '{query_text_or_id}' was not found. Showing top PageRank paper '{target_pid}']\n\n"
        elif note:
            header_note = f"[GraphRAG Note: {note}]\n\n"

        # Construct Rich GraphRAG Prompt Block
        prompt_block = (
            f"{header_note}"
            f"You are an AI research assistant. Use the following citation knowledge graph context to answer questions about the research paper.\n\n"
            f"Paper:\n{title}\n\n"
            f"Paper ID:\n{target_pid}\n\n"
            f"Authors:\n{', '.join(authors) if authors else 'N/A'}\n\n"
            f"Category:\n{category}\n\n"
            f"Citations (Papers citing this work):\n"
            f"{chr(10).join(['- ' + p for p in cited_by_papers]) if cited_by_papers else 'None registered'}\n\n"
            f"References (Papers cited by this work):\n"
            f"{chr(10).join(['- ' + p for p in cited_papers]) if cited_papers else 'None registered'}\n\n"
            f"Related Papers (Same Community #{comm_id}):\n"
            f"{chr(10).join(['- ' + p for p in related_community_papers]) if related_community_papers else 'None'}\n\n"
            f"Graph Relationships:\n"
            f"{chr(10).join(graph_relationships[:8]) if graph_relationships else 'N/A'}\n\n"
            f"PageRank Authority Score: {pr_score:.6f}\n\n"
            f"Use only the provided graph context when making structural claims."
        )

        return {
            "target_paper_id": target_pid,
            "title": title,
            "category": category,
            "year": year,
            "authors": authors,
            "topics": topics,
            "cited_references": cited_papers,
            "cited_by_citations": cited_by_papers,
            "related_community_papers": related_community_papers,
            "graph_relationships": graph_relationships,
            "pagerank_prestige": pr_score,
            "research_community": comm_id,
            "graphrag_prompt_block": prompt_block
        }

# Global singleton helper for reliable loader
_ckg_singleton = None

def get_or_build_citational_graph(max_papers=2500):
    global _ckg_singleton
    if _ckg_singleton is not None:
        return _ckg_singleton

    ckg = CitationalKnowledgeGraph()
    ckg.load_data()
    ckg.build_graph(max_papers=max_papers)
    ckg.compute_structural_analysis()
    ckg.visualize_graph()
    _ckg_singleton = ckg
    return ckg

# ---------------------------------------------------------
# EXECUTION ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("OBJECTIVE 1: CITATIONAL KNOWLEDGE GRAPH (GraphRAG) PIPELINE")
    print("=" * 70)
    
    ckg = get_or_build_citational_graph(max_papers=2500)
    
    # Save instance
    joblib.dump(ckg, os.path.join(MODELS_DIR, "citational_knowledge_graph.pkl"))
    
    # Run test retrievals
    print("\n[5/5] Testing GraphRAG Context Retrieval...")
    print("--- Test 1: Short ID 'P_001' ---")
    t1 = ckg.retrieve_graphrag_context("P_001")
    print(t1["graphrag_prompt_block"])
    
    print("\n--- Test 2: Title search 'Deep Learning' ---")
    t2 = ckg.retrieve_graphrag_context("Deep Learning")
    print(t2["graphrag_prompt_block"])
    
    print("\n[SUCCESS] Objective 1 Knowledge Graph Pipeline Completed Successfully!")
