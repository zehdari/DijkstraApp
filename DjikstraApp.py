import sys
import math
import heapq
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QInputDialog, QMessageBox, QComboBox, QDialog, QVBoxLayout, QLabel, QPushButton, QToolBar,
    QFileDialog
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath, QIcon, QAction
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer

class NodeSelectionDialog(QDialog):
    def __init__(self, node_labels):
        super().__init__()
        self.setWindowTitle("Select Start and End Nodes")
        self.start_node = None
        self.end_node = None
        layout = QVBoxLayout()

        self.start_combo = QComboBox()
        self.start_combo.addItems(node_labels)
        layout.addWidget(QLabel("Select Start Node:"))
        layout.addWidget(self.start_combo)

        self.end_combo = QComboBox()
        self.end_combo.addItems(node_labels)
        layout.addWidget(QLabel("Select End Node:"))
        layout.addWidget(self.end_combo)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

        self.setLayout(layout)

    def accept(self):
        self.start_node = self.start_combo.currentIndex()
        self.end_node = self.end_combo.currentIndex()
        super().accept()

class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.nodes = []  
        self.edges = [] 
        self.node_labels = []  
        self.selected_node = None  
        self.dragging_node = None 
        self.current_mouse_pos = None  
        self.offset = QPointF(0, 0)  
        self.setMouseTracking(True)
        self.is_dragging = False  
        self.mouse_press_pos = None  
        self.drag_threshold = 5  
        self.edge_weight_positions = []  

        # Dijkstra's algorithm variables
        self.dijkstra_running = False
        self.dijkstra_timer = QTimer()
        self.dijkstra_timer.timeout.connect(self.dijkstra_step)
        self.dijkstra_data = None  
        self.visited_nodes = set()
        self.current_node = None
        self.shortest_path = []
        self.highlighted_edges = set()
        self.node_distances = {}  # To store tentative distances for visualization

    def run_dijkstra(self, start_node, end_node):
        # Initialize Dijkstra's algorithm data
        num_nodes = len(self.nodes)
        distances = [math.inf] * num_nodes
        previous = [None] * num_nodes
        distances[start_node] = 0
        queue = [(0, start_node)]
        self.dijkstra_data = {
            'distances': distances,
            'previous': previous,
            'queue': queue,
            'end_node': end_node
        }
        self.visited_nodes = set()
        self.current_node = None
        self.shortest_path = []
        self.highlighted_edges = set()
        self.node_distances = {}  # Reset distances
        self.dijkstra_running = True
        self.dijkstra_timer.start(1500)  # Step every 1500 milliseconds (1.5 seconds)
        self.update()

    def dijkstra_step(self):
        if not self.dijkstra_data['queue']:
            self.dijkstra_timer.stop()
            self.dijkstra_running = False
            self.extract_shortest_path()
            self.update()
            QMessageBox.information(self, "Dijkstra's Algorithm", "No path found.")
            return

        distances = self.dijkstra_data['distances']
        previous = self.dijkstra_data['previous']
        queue = self.dijkstra_data['queue']
        end_node = self.dijkstra_data['end_node']

        current_distance, current_node = heapq.heappop(queue)
        if current_node in self.visited_nodes:
            return

        self.visited_nodes.add(current_node)
        self.current_node = current_node
        self.update_node_distances(distances)
        self.update()

        if current_node == end_node:
            self.dijkstra_timer.stop()
            self.dijkstra_running = False
            self.extract_shortest_path()
            self.update()
            QMessageBox.information(self, "Dijkstra's Algorithm", "Shortest path found.")
            return

        # Get neighbors
        neighbors = []
        for start_idx, end_idx, weight in self.edges:
            if start_idx == current_node and end_idx not in self.visited_nodes:
                neighbors.append((end_idx, weight))

        for neighbor, weight in neighbors:
            alt_distance = current_distance + weight
            if alt_distance < distances[neighbor]:
                distances[neighbor] = alt_distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (alt_distance, neighbor))
                # Highlight the edge being relaxed
                self.highlighted_edges.add((current_node, neighbor))
                self.update_node_distances(distances)
                self.update()
                QTimer.singleShot(1000, lambda s=current_node, n=neighbor: self.remove_highlighted_edge(s, n))

    def remove_highlighted_edge(self, start_node, end_node):
        self.highlighted_edges.discard((start_node, end_node))
        self.update()

    def extract_shortest_path(self):
        # Extract the shortest path from the 'previous' array
        previous = self.dijkstra_data['previous']
        end_node = self.dijkstra_data['end_node']
        node = end_node
        path = []
        while node is not None:
            path.insert(0, node)
            node = previous[node]
        self.shortest_path = path

    def update_node_distances(self, distances):
        self.node_distances = {i: distances[i] for i in range(len(distances)) if distances[i] != math.inf}

    def reset_dijkstra_visualization(self):
        # Reset all Dijkstra's algorithm related variables and visualizations
        self.dijkstra_running = False
        self.dijkstra_timer.stop()
        self.dijkstra_data = None
        self.visited_nodes = set()
        self.current_node = None
        self.shortest_path = []
        self.highlighted_edges = set()
        self.node_distances = {}
        self.update()

    def mousePressEvent(self, event):
        if self.dijkstra_running:
            return  # Ignore input during algorithm execution

        # Reset Dijkstra's algorithm visualization on any user interaction
        self.reset_dijkstra_visualization()

        pos = event.position()
        node_index = self.get_node_at_position(pos)
        modifiers = QApplication.keyboardModifiers()

        # First, check if click is near any edge weight text to edit weight
        for i, (start_index, end_index, text_rect) in enumerate(self.edge_weight_positions):
            if text_rect.contains(pos):
                # Prompt user to enter new weight
                for edge_i, (s_idx, e_idx, weight) in enumerate(self.edges):
                    if s_idx == start_index and e_idx == end_index:
                        new_weight, ok = QInputDialog.getInt(self, "Edit Edge Weight", "Enter new weight:", value=weight)
                        if ok:
                            # Update the weight
                            self.edges[edge_i] = (s_idx, e_idx, new_weight)
                            self.update()
                        return  

        # Then, check if click is near any arrowhead to reverse edge direction
        for i, (start_index, end_index, weight) in enumerate(self.edges):
            start_pos = self.nodes[start_index]
            end_pos = self.nodes[end_index]
            reverse_exists = self.edge_exists(end_index, start_index)
            if self.is_click_near_arrowhead(pos, start_pos, end_pos):
                if reverse_exists:
                    # Disable arrowhead flip to prevent duplicate edges
                    pass
                else:
                    # Flip the direction of the edge
                    self.edges[i] = (end_index, start_index, weight)
                    self.update()
                return  

        if event.button() == Qt.MouseButton.LeftButton:
            if node_index is not None:
                self.mouse_press_pos = pos
                self.dragging_node = node_index
                self.offset = self.nodes[node_index] - pos
                self.is_dragging = False 
            else:
                if self.selected_node is not None:
                    # Clicked on background; deselect the node
                    self.selected_node = None
                    self.current_mouse_pos = None
                    self.update()
                else:
                    if modifiers == Qt.KeyboardModifier.ShiftModifier:
                        # Create a new node
                        self.nodes.append(pos)
                        self.update_node_labels()
                        self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if node_index is not None:
                self.remove_node(node_index)
                self.update_node_labels()
                self.update()
            else:
                # Check if click is on an edge to remove it
                edge_index = self.get_edge_at_position(pos)
                if edge_index is not None:
                    self.remove_edge(edge_index)
                    self.update()

    def mouseMoveEvent(self, event):
        if self.dijkstra_running:
            return  # Ignore input during algorithm execution

        if self.dragging_node is not None or self.selected_node is not None:
            # Reset Dijkstra's algorithm visualization on any user interaction
            self.reset_dijkstra_visualization()

        pos = event.position()
        if self.dragging_node is not None:
            # Check if the mouse has moved beyond the drag threshold
            if not self.is_dragging and (pos - self.mouse_press_pos).manhattanLength() > self.drag_threshold:
                self.is_dragging = True  
            if self.is_dragging:
                self.nodes[self.dragging_node] = pos + self.offset
                self.update()
        elif self.selected_node is not None:
            self.current_mouse_pos = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dijkstra_running:
            return  # Ignore input during algorithm execution

        pos = event.position()
        node_index = self.get_node_at_position(pos)
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging_node is not None:
                if not self.is_dragging:
                    # Treat as a click since no dragging occurred
                    if self.selected_node is not None:
                        if node_index != self.selected_node and node_index is not None:
                            # Check if edge already exists
                            if not self.edge_exists(self.selected_node, node_index):
                                # Add edge between selected nodes with default weight 1
                                self.edges.append((self.selected_node, node_index, 1))
                                self.update()
                            else:
                                # Edge already exists
                                pass
                            self.selected_node = None
                            self.current_mouse_pos = None
                        else:
                            # No edge created, keep selected_node for further actions
                            pass
                    else:
                        # Select the node for edge drawing
                        self.selected_node = self.dragging_node
                        self.current_mouse_pos = pos
                # Reset dragging variables
                self.dragging_node = None
                self.is_dragging = False
            else:
                if self.selected_node is not None and node_index == self.selected_node:
                    # Deselect the node if it's clicked again
                    self.selected_node = None
                    self.current_mouse_pos = None
                    self.update()
        self.mouse_press_pos = None  # Reset mouse press position

    def edge_exists(self, start, end):
        return any((s == start and e == end) for s, e, _ in self.edges)

    def get_node_at_position(self, pos):
        for index, node_pos in enumerate(self.nodes):
            if (node_pos - pos).manhattanLength() < 15:  # Node radius threshold
                return index
        return None

    def remove_node(self, index):
        del self.nodes[index]
        # Remove edges associated with this node
        self.edges = [
            (start, end, weight)
            for start, end, weight in self.edges
            if start != index and end != index
        ]
        # Adjust indices in edges
        self.edges = [
            (
                start - (1 if start > index else 0),
                end - (1 if end > index else 0),
                weight
            )
            for start, end, weight in self.edges
        ]
        self.update_node_labels()

    def remove_edge(self, index):
        del self.edges[index]

    def update_node_labels(self):
        self.node_labels = [chr(ord('A') + i) for i in range(len(self.nodes))]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor("#2C3E50"), 2)
        painter.setPen(pen)
        self.edge_weight_positions = []  # Reset edge weight positions

        # Build a set of edge pairs for quick lookup
        edge_pairs = set()
        for start_index, end_index, _ in self.edges:
            edge_pairs.add((start_index, end_index))

        # Draw edges
        for start_index, end_index, weight in self.edges:
            start_pos = self.nodes[start_index]
            end_pos = self.nodes[end_index]

            reverse_exists = (end_index, start_index) in edge_pairs

            if reverse_exists and start_index != end_index:
                # Draw curved edge
                self.draw_curved_edge(painter, start_index, end_index, weight)
            else:
                # Draw straight edge
                self.draw_straight_edge(painter, start_index, end_index, weight)

        # Draw the edge being created
        if self.selected_node is not None and self.current_mouse_pos is not None:
            start_pos = self.nodes[self.selected_node]
            end_pos = self.current_mouse_pos

            # Adjust positions to be at the edge of the node
            adjusted_start, _ = self.get_line_points(start_pos, end_pos)

            painter.setPen(QPen(QColor("#2C3E50"), 2, Qt.PenStyle.DashLine))
            painter.drawLine(adjusted_start, end_pos)
            painter.setPen(pen)

        # Draw nodes
        for index, pos in enumerate(self.nodes):
            painter.setPen(QPen(QColor("#2980B9"), 1))
            if self.dijkstra_running:
                if index == self.current_node:
                    painter.setBrush(QBrush(QColor("#F1C40F")))  # Current node
                elif index in self.visited_nodes:
                    painter.setBrush(QBrush(QColor("#95A5A6")))  # Visited nodes
                else:
                    painter.setBrush(QBrush(QColor("#3498DB")))
            else:
                if index == self.selected_node:
                    painter.setBrush(QBrush(QColor("#F1C40F")))  # Highlight selected node
                else:
                    painter.setBrush(QBrush(QColor("#3498DB")))

            # Nodes in shortest path are always green
            if index in self.shortest_path:
                painter.setBrush(QBrush(QColor("#27AE60")))  # Nodes in shortest path

            painter.drawEllipse(pos, 15, 15)  # Node radius 15

            # Draw node label and tentative distance
            painter.setPen(Qt.GlobalColor.black)
            label = self.node_labels[index]
            if self.dijkstra_running and index in self.node_distances:
                distance = self.node_distances[index]
                label += f"\n({distance})"
            elif index in self.node_distances and not self.dijkstra_running:
                # Show final distances after algorithm completes
                distance = self.node_distances[index]
                label += f"\n({distance})"
            painter.drawText(pos + QPointF(-10, 5), label)
            painter.setPen(QPen(QColor("#2980B9"), 1))

    def draw_straight_edge(self, painter, start_index, end_index, weight):
        start_pos = self.nodes[start_index]
        end_pos = self.nodes[end_index]

        adjusted_start, adjusted_end = self.get_line_points(start_pos, end_pos)

        # Highlight edge if part of the shortest path or being relaxed
        if (start_index, end_index) in self.highlighted_edges:
            painter.setPen(QPen(QColor("#E67E22"), 3))
        elif self.is_edge_in_shortest_path(start_index, end_index):
            painter.setPen(QPen(QColor("#27AE60"), 3))
        else:
            painter.setPen(QPen(QColor("#2C3E50"), 2))

        painter.drawLine(adjusted_start, adjusted_end)

        # Draw arrowhead
        angle = math.atan2(adjusted_end.y() - adjusted_start.y(), adjusted_end.x() - adjusted_start.x())
        self.draw_arrowhead(painter, adjusted_end, angle)

        # Draw weight text at midpoint
        mid_point = (adjusted_start + adjusted_end) / 2
        painter.setPen(QColor("#E74C3C"))
        text = f"{weight}"
        painter.drawText(mid_point, text)
        # Get the bounding rectangle
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()
        text_rect = QRectF(mid_point.x() - text_width / 2, mid_point.y() - text_height / 2, text_width, text_height)
        # Expand the clickable area
        padding = 5 
        expanded_rect = text_rect.adjusted(-padding, -padding, padding, padding)
        # Store the position
        self.edge_weight_positions.append((start_index, end_index, expanded_rect))
        painter.setPen(QPen(QColor("#2C3E50"), 2))

    def draw_curved_edge(self, painter, start_index, end_index, weight):
        start_pos = self.nodes[start_index]
        end_pos = self.nodes[end_index]

        # Compute control point for curve
        dx = end_pos.x() - start_pos.x()
        dy = end_pos.y() - start_pos.y()
        center = (start_pos + end_pos) / 2
        # Perpendicular vector
        perp = QPointF(-dy, dx)
        perp = perp / math.hypot(perp.x(), perp.y())
        offset = perp * 30  # Adjust offset as needed
        control_point = center + offset

        # Create path for curved edge
        path = QPainterPath()
        adjusted_start, adjusted_end = self.get_line_points(start_pos, end_pos, for_curve=True)
        path.moveTo(adjusted_start)
        path.quadTo(control_point, adjusted_end)

        # Highlight edge if part of the shortest path or being relaxed
        if (start_index, end_index) in self.highlighted_edges:
            painter.setPen(QPen(QColor("#E67E22"), 3))
        elif self.is_edge_in_shortest_path(start_index, end_index):
            painter.setPen(QPen(QColor("#27AE60"), 3))
        else:
            painter.setPen(QPen(QColor("#2C3E50"), 2))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Draw arrowhead
        # Approximate angle at end point
        t = 0.95  # Parameter near end of curve
        point_tangent = self.quadratic_bezier_tangent(adjusted_start, control_point, adjusted_end, t)
        angle = math.atan2(point_tangent.y(), point_tangent.x())
        self.draw_arrowhead(painter, adjusted_end, angle)

        # Draw weight text at control point
        painter.setPen(QColor("#E74C3C"))
        text = f"{weight}"
        painter.drawText(control_point, text)
        # Get the bounding rectangle
        font_metrics = painter.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        text_height = font_metrics.height()
        text_rect = QRectF(control_point.x() - text_width / 2, control_point.y() - text_height / 2, text_width, text_height)
        # Expand the clickable area
        padding = 5 
        expanded_rect = text_rect.adjusted(-padding, -padding, padding, padding)
        # Store the position
        self.edge_weight_positions.append((start_index, end_index, expanded_rect))
        painter.setPen(QPen(QColor("#2C3E50"), 2))

    def is_edge_in_shortest_path(self, start_index, end_index):
        # Check if the edge is part of the shortest path
        for i in range(len(self.shortest_path) - 1):
            if self.shortest_path[i] == start_index and self.shortest_path[i + 1] == end_index:
                return True
        return False

    def quadratic_bezier_point(self, p0, p1, p2, t):
        x = (1 - t)**2 * p0.x() + 2 * (1 - t) * t * p1.x() + t**2 * p2.x()
        y = (1 - t)**2 * p0.y() + 2 * (1 - t) * t * p1.y() + t**2 * p2.y()
        return QPointF(x, y)

    def quadratic_bezier_tangent(self, p0, p1, p2, t):
        # Derivative of quadratic Bezier curve
        x = 2*(1 - t)*(p1.x() - p0.x()) + 2*t*(p2.x() - p1.x())
        y = 2*(1 - t)*(p1.y() - p0.y()) + 2*t*(p2.y() - p1.y())
        return QPointF(x, y)

    def draw_arrowhead(self, painter, end_pos, angle):
        arrow_size = 10

        # Points for the arrowhead
        arrow_p1 = QPointF(
            end_pos.x() - arrow_size * math.cos(angle - math.pi / 6),
            end_pos.y() - arrow_size * math.sin(angle - math.pi / 6)
        )
        arrow_p2 = QPointF(
            end_pos.x() - arrow_size * math.cos(angle + math.pi / 6),
            end_pos.y() - arrow_size * math.sin(angle + math.pi / 6)
        )
        # Draw the arrowhead
        arrow_head = QPolygonF([end_pos, arrow_p1, arrow_p2])
        painter.setBrush(QBrush(QColor("#2C3E50")))
        painter.setPen(Qt.PenStyle.NoPen)  # No border for the arrowhead
        painter.drawPolygon(arrow_head)
        painter.setPen(QPen(QColor("#2C3E50"), 2)) 

    def is_click_near_arrowhead(self, click_pos, start_pos, end_pos):
        arrow_size = 10
        adjusted_start, adjusted_end = self.get_line_points(start_pos, end_pos)
        arrow_point = adjusted_end
        # Check if click is near the arrowhead
        distance = (click_pos - arrow_point).manhattanLength()
        return distance < arrow_size

    def get_line_points(self, start_pos, end_pos, for_curve=False):
        # Adjust positions to be at the edge of the nodes
        dx = end_pos.x() - start_pos.x()
        dy = end_pos.y() - start_pos.y()
        line_length = math.hypot(dx, dy)
        if line_length == 0:
            return start_pos, end_pos  # Avoid division by zero
        unit_dx = dx / line_length
        unit_dy = dy / line_length

        adjust_start = 15
        adjust_end = 15

        start_x = start_pos.x() + unit_dx * adjust_start 
        start_y = start_pos.y() + unit_dy * adjust_start
        end_x = end_pos.x() - unit_dx * adjust_end  
        end_y = end_pos.y() - unit_dy * adjust_end

        adjusted_start = QPointF(start_x, start_y)
        adjusted_end = QPointF(end_x, end_y)
        return adjusted_start, adjusted_end

    def get_edge_at_position(self, pos):
        # Iterate over edges
        for i, (start_index, end_index, weight) in enumerate(self.edges):
            start_pos = self.nodes[start_index]
            end_pos = self.nodes[end_index]
            reverse_exists = self.edge_exists(end_index, start_index)
            if reverse_exists and start_index != end_index:
                # For curved edges
                if self.is_click_near_curved_edge(pos, start_pos, end_pos):
                    return i
            else:
                # For straight edges
                if self.is_click_near_straight_edge(pos, start_pos, end_pos):
                    return i
        return None

    def is_click_near_straight_edge(self, click_pos, start_pos, end_pos):
        threshold = 5.0
        # Adjust positions to be at the edge of the nodes
        adjusted_start, adjusted_end = self.get_line_points(start_pos, end_pos)
        # Compute distance from click_pos to the line segment adjusted_start - adjusted_end
        line_vec = adjusted_end - adjusted_start
        line_len = math.hypot(line_vec.x(), line_vec.y())
        if line_len == 0:
            return False
        line_unitvec = line_vec / line_len
        point_vec = click_pos - adjusted_start
        proj_length = point_vec.x() * line_unitvec.x() + point_vec.y() * line_unitvec.y()
        if proj_length < 0 or proj_length > line_len:
            return False  # Click is outside the line segment
        proj_point = adjusted_start + line_unitvec * proj_length
        distance = (click_pos - proj_point).manhattanLength()
        if distance <= threshold:
            return True
        else:
            return False

    def is_click_near_curved_edge(self, click_pos, start_pos, end_pos):
        threshold = 5.0 
        # Compute control point for curve
        dx = end_pos.x() - start_pos.x()
        dy = end_pos.y() - start_pos.y()
        center = (start_pos + end_pos) / 2
        # Perpendicular vector
        perp = QPointF(-dy, dx)
        perp = perp / math.hypot(perp.x(), perp.y())
        offset = perp * 30  # Use same offset as in draw_curved_edge
        control_point = center + offset

        # Adjust positions to be at the edge of the nodes
        adjusted_start, adjusted_end = self.get_line_points(start_pos, end_pos, for_curve=True)

        # Sample points along the curve
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            point = self.quadratic_bezier_point(adjusted_start, control_point, adjusted_end, t)
            distance = (click_pos - point).manhattanLength()
            if distance <= threshold:
                return True
        return False

    def save_graph(self, filename):
        # Prepare data for serialization
        nodes = [{'x': pos.x(), 'y': pos.y()} for pos in self.nodes]
        edges = [{'start': start, 'end': end, 'weight': weight} for start, end, weight in self.edges]
        data = {
            'nodes': nodes,
            'edges': edges,
            'node_labels': self.node_labels
        }
        # Write data to file
        with open(filename, 'w') as f:
            json.dump(data, f)
        QMessageBox.information(self, "Save Graph", f"Graph saved successfully to {filename}.")

    def load_graph(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.load_graph_data(data)
            QMessageBox.information(self, "Load Graph", f"Graph loaded successfully from {filename}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load graph: {e}")

    def load_graph_data(self, data):
        try:
            # Load nodes
            self.nodes = [QPointF(node['x'], node['y']) for node in data['nodes']]
            # Load edges
            self.edges = [(edge['start'], edge['end'], edge['weight']) for edge in data['edges']]
            # Load node labels
            self.node_labels = data.get('node_labels', [])
            # Reset algorithm variables
            self.selected_node = None
            self.dragging_node = None
            self.current_mouse_pos = None
            self.is_dragging = False
            self.mouse_press_pos = None
            self.edge_weight_positions = []
            self.reset_dijkstra_visualization()
            self.update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load graph data: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DjikstraApp")
        self.setGeometry(100, 100, 800, 600)
        self.graph_widget = GraphWidget()
        self.setCentralWidget(self.graph_widget)
        self.apply_styles()
        self.create_actions()
        self.create_toolbar()
        self.load_default_graph()

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ECF0F1;
            }
            QWidget {
                font-size: 14px;
            }
        """)

    def create_actions(self):
        self.run_dijkstra_action = QAction("Run Dijkstra's Algorithm", self)
        self.run_dijkstra_action.triggered.connect(self.run_dijkstra)

        self.save_graph_action = QAction("Save Graph", self)
        self.save_graph_action.triggered.connect(self.save_graph)

        self.load_graph_action = QAction("Load Graph", self)
        self.load_graph_action.triggered.connect(self.load_graph)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(self.run_dijkstra_action)
        toolbar.addAction(self.save_graph_action)
        toolbar.addAction(self.load_graph_action)

    def run_dijkstra(self):
        if len(self.graph_widget.nodes) < 2:
            QMessageBox.warning(self, "Warning", "Need at least two nodes to run Dijkstra's algorithm.")
            return

        dialog = NodeSelectionDialog(self.graph_widget.node_labels)
        if dialog.exec():
            start_node = dialog.start_node
            end_node = dialog.end_node
            if start_node == end_node:
                QMessageBox.warning(self, "Warning", "Start and end nodes must be different.")
                return
            self.graph_widget.run_dijkstra(start_node, end_node)

    def save_graph(self):
        options = QFileDialog.Option.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", "Graph Files (*.json);;All Files (*)", options=options)
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            self.graph_widget.save_graph(filename)

    def load_graph(self):
        options = QFileDialog.Option.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Graph", "", "Graph Files (*.json);;All Files (*)", options=options)
        if filename:
            self.graph_widget.load_graph(filename)

    def load_default_graph(self):
        default_graph_data = {
            "nodes": [
                {"x": 100.0, "y": 100.0},
                {"x": 200.0, "y": 50.0},
                {"x": 300.0, "y": 100.0},
                {"x": 100.0, "y": 200.0},
                {"x": 200.0, "y": 250.0},
                {"x": 300.0, "y": 200.0},
                {"x": 200.0, "y": 150.0}
            ],
            "edges": [
                {"start": 0, "end": 1, "weight": 2},
                {"start": 0, "end": 3, "weight": 1},
                {"start": 1, "end": 2, "weight": 3},
                {"start": 2, "end": 5, "weight": 2},
                {"start": 5, "end": 4, "weight": 1},
                {"start": 3, "end": 4, "weight": 4},
                {"start": 4, "end": 6, "weight": 2},
                {"start": 6, "end": 1, "weight": 1},
                {"start": 6, "end": 2, "weight": 2},
                {"start": 6, "end": 5, "weight": 3}
            ],
            "node_labels": ["A", "B", "C", "D", "E", "F", "G"]
        }
        self.graph_widget.load_graph_data(default_graph_data)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
