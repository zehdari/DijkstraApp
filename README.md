
# DjikstraApp - Graph Creator and Dijkstra's Algorithm Visualizer

**DjikstraApp** is an interactive PyQt-based application for creating graphs and visualizing Dijkstra's algorithm. It allows users to build custom graphs, edit nodes and edges, and step through Dijkstra's algorithm to find the shortest path between nodes in real-time.

## Features

- **Interactive Graph Creation**:
  - Add, move, and delete nodes with ease.
  - Create directed edges with custom weights.
  - Edit edge weights and reverse edge directions interactively.
  - Right-click to delete nodes or edges.
  - Support for both straight and curved edges for bidirectional connections.
- **Dijkstra's Algorithm Visualization**:
  - Step-by-step execution with visual highlights for current nodes, tentative distances, and visited nodes.
  - Visual representation of the shortest path upon completion.
  - Adjustable speed of algorithm execution.
- **Graph Management**:
  - Save and load custom graphs in JSON format.
  - Intuitive UI with toolbars and dialogs for user interactions.

## Installation

### Precompiled Executables

You can download precompiled executables for **macOS**, **Ubuntu**, **Windows**, and **RedHat** from the [Releases](https://github.com/zehdari/DjikstraApp/releases) page.

1. **Download** the appropriate zip file for your operating system:
   - macOS
   - Ubuntu
   - Windows
   - RedHat

2. **Extract the zip file**:
   - On macOS and Linux:

     ```bash
     unzip DjikstraApp-<os>.zip
     cd DjikstraApp
     ```

   - On Windows:
     - Right-click the zip file and choose "Extract All".
     - Open the extracted folder.

3. **Run the executable**:
   - On macOS and Ubuntu:

     ```bash
     ./DjikstraApp
     ```

   - On Windows:
     - Double-click `DjikstraApp.exe`.
   - On RedHat:
     After unzipping, you need to make the file executable:

     ```bash
     chmod +x DjikstraApp
     ./DjikstraApp
     ```

### From Source

If you prefer to run the app from the source code, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/zehdari/DjikstraApp
   cd DjikstraApp
   ```

2. **Set up a virtual environment**:
   Ensure you are using Python 3. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   Install the required dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   python DjikstraApp.py
   ```

## Usage

### Creating Nodes and Edges:

- **Add Node**: Shift + Left-click on the canvas to add a new node.
- **Move Node**: Left-click and drag a node to reposition it.
- **Delete Node**: Right-click on a node to delete it.
- **Create Edge**: Left-click on a node to select it, then click on another node to create a directed edge.
- **Delete Edge**: Right-click near an edge to delete it.
- **Edit Edge Weight**: Click on the edge weight label to change its value.
- **Reverse Edge Direction**: Click near the arrowhead of an edge to flip its direction.

### Running Dijkstra's Algorithm:

1. Click on the "Run Dijkstra's Algorithm" button in the toolbar.
2. Select the start and end nodes from the dialog.
3. The algorithm will execute step-by-step, highlighting:
   - Current node being explored.
   - Tentative distances to each node.
   - Visited nodes.
   - Edges being relaxed.
4. Upon completion, the shortest path will be highlighted.

### Saving and Loading Graphs:

- **Save Graph**: Click on the "Save Graph" button to save your current graph to a JSON file.
- **Load Graph**: Click on the "Load Graph" button to load a previously saved graph.

## Controls

### Toolbar Buttons:

- **Run Dijkstra's Algorithm**: Opens a dialog to select start and end nodes.
- **Save Graph**: Saves the current graph to a file.
- **Load Graph**: Loads a graph from a file.

### Mouse Interactions:

- **Shift + Left-click**: Add a new node at the cursor position.
- **Left-click on Node**: Selects a node for edge creation or deselects if clicked again.
- **Left-click and Drag**: Move a node to a new position.
- **Right-click on Node**: Delete the selected node.
- **Right-click near Edge**: Delete the selected edge.
- **Click on Edge Weight**: Edit the weight of the edge.
- **Click near Edge Arrowhead**: Reverse the direction of the edge.

## File Structure

- **`DjikstraApp.py`**: Main file containing the DjikstraApp UI and logic for graph creation and algorithm visualization.
- **`resources/`**: Folder containing icons and other static resources for the application.
- **`requirements.txt`**: Python dependencies required to run DjikstraApp from the source.
