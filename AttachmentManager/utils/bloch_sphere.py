import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Tuple, Dict, Any
import plotly.express as px

class BlochSphere:
    """
    Bloch sphere visualization utility for quantum state representation.
    Creates interactive 3D visualizations of single qubit quantum states.
    """
    
    def __init__(self):
        """Initialize the Bloch sphere visualization utility."""
        # Sphere parameters
        self.sphere_resolution = 50
        self.vector_scale = 1.0
        self.axis_length = 1.2
        
        # Color scheme
        self.sphere_color = 'lightblue'
        self.sphere_opacity = 0.3
        self.vector_color = 'red'
        self.axis_color = 'black'
        self.equator_color = 'blue'
        self.meridian_color = 'green'
        
        # Labels
        self.axis_labels = {
            'x_pos': '+X',
            'x_neg': '-X',
            'y_pos': '+Y',
            'y_neg': '-Y',
            'z_pos': '|0⟩',
            'z_neg': '|1⟩'
        }
    
    def create_bloch_sphere(self, theta: float, phi: float, 
                           show_axes: bool = True,
                           show_equator: bool = True,
                           show_meridians: bool = True,
                           title: str = "Bloch Sphere") -> go.Figure:
        """
        Create a complete Bloch sphere visualization.
        
        Args:
            theta: Polar angle (0 to π)
            phi: Azimuthal angle (0 to 2π)
            show_axes: Whether to show coordinate axes
            show_equator: Whether to show equatorial circle
            show_meridians: Whether to show meridian lines
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add the sphere surface
        fig.add_trace(self._create_sphere_surface())
        
        # Add coordinate axes
        if show_axes:
            self._add_coordinate_axes(fig)
        
        # Add equator and meridians
        if show_equator:
            self._add_equator(fig)
        
        if show_meridians:
            self._add_meridians(fig)
        
        # Add state vector
        self._add_state_vector(fig, theta, phi)
        
        # Add labels
        self._add_axis_labels(fig)
        
        # Configure layout
        self._configure_layout(fig, title)
        
        return fig
    
    def create_multiple_states_sphere(self, states: List[Tuple[float, float]], 
                                    labels: Optional[List[str]] = None,
                                    colors: Optional[List[str]] = None,
                                    title: str = "Multiple Quantum States") -> go.Figure:
        """
        Create Bloch sphere with multiple quantum states.
        
        Args:
            states: List of (theta, phi) tuples for each state
            labels: Optional labels for each state
            colors: Optional colors for each state vector
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add the sphere surface
        fig.add_trace(self._create_sphere_surface())
        
        # Add coordinate axes and grid
        self._add_coordinate_axes(fig)
        self._add_equator(fig)
        self._add_meridians(fig)
        
        # Default colors if not provided
        if colors is None:
            colors = px.colors.qualitative.Set1[:len(states)]
        
        # Add each state vector
        for i, (theta, phi) in enumerate(states):
            color = colors[i % len(colors)]
            label = labels[i] if labels else f"State {i+1}"
            self._add_state_vector(fig, theta, phi, color=color, name=label)
        
        # Add labels
        self._add_axis_labels(fig)
        
        # Configure layout
        self._configure_layout(fig, title)
        
        return fig
    
    def create_evolution_trajectory(self, trajectory: List[Tuple[float, float]],
                                  time_points: Optional[List[float]] = None,
                                  title: str = "State Evolution") -> go.Figure:
        """
        Create Bloch sphere showing state evolution trajectory.
        
        Args:
            trajectory: List of (theta, phi) points along evolution path
            time_points: Optional time points for trajectory
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add the sphere surface
        fig.add_trace(self._create_sphere_surface())
        
        # Add coordinate system
        self._add_coordinate_axes(fig)
        self._add_equator(fig)
        self._add_meridians(fig)
        
        # Convert trajectory to Cartesian coordinates
        x_traj, y_traj, z_traj = zip(*[self._spherical_to_cartesian(theta, phi) 
                                      for theta, phi in trajectory])
        
        # Add trajectory path
        fig.add_trace(go.Scatter3d(
            x=x_traj,
            y=y_traj,
            z=z_traj,
            mode='lines+markers',
            line=dict(width=4, color='red'),
            marker=dict(size=3, color='red'),
            name='Evolution Path',
            hovertemplate='<b>Evolution Path</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        ))
        
        # Highlight start and end points
        if len(trajectory) > 0:
            # Start point
            start_x, start_y, start_z = self._spherical_to_cartesian(trajectory[0][0], trajectory[0][1])
            fig.add_trace(go.Scatter3d(
                x=[start_x],
                y=[start_y],
                z=[start_z],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name='Initial State',
                hovertemplate='<b>Initial State</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>'
            ))
            
            # End point
            end_x, end_y, end_z = self._spherical_to_cartesian(trajectory[-1][0], trajectory[-1][1])
            fig.add_trace(go.Scatter3d(
                x=[end_x],
                y=[end_y],
                z=[end_z],
                mode='markers',
                marker=dict(size=8, color='blue', symbol='diamond'),
                name='Final State',
                hovertemplate='<b>Final State</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>'
            ))
        
        # Add labels
        self._add_axis_labels(fig)
        
        # Configure layout
        self._configure_layout(fig, title)
        
        return fig
    
    def create_measurement_visualization(self, theta: float, phi: float,
                                       measurement_basis: str = 'Z',
                                       show_probabilities: bool = True,
                                       title: str = "Measurement Visualization") -> go.Figure:
        """
        Create Bloch sphere showing measurement projections.
        
        Args:
            theta: Polar angle of quantum state
            phi: Azimuthal angle of quantum state
            measurement_basis: Measurement basis ('X', 'Y', or 'Z')
            show_probabilities: Whether to show probability annotations
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add the sphere surface
        fig.add_trace(self._create_sphere_surface())
        
        # Add coordinate system
        self._add_coordinate_axes(fig)
        self._add_equator(fig)
        self._add_meridians(fig)
        
        # Add state vector
        self._add_state_vector(fig, theta, phi)
        
        # Add measurement basis visualization
        self._add_measurement_basis(fig, measurement_basis)
        
        # Calculate and show projections
        if show_probabilities:
            self._add_measurement_projections(fig, theta, phi, measurement_basis)
        
        # Add labels
        self._add_axis_labels(fig)
        
        # Configure layout
        self._configure_layout(fig, title)
        
        return fig
    
    def _create_sphere_surface(self) -> go.Surface:
        """Create the Bloch sphere surface."""
        # Generate sphere coordinates
        u = np.linspace(0, 2 * np.pi, self.sphere_resolution)
        v = np.linspace(0, np.pi, self.sphere_resolution)
        
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        return go.Surface(
            x=x, y=y, z=z,
            opacity=self.sphere_opacity,
            colorscale=[[0, self.sphere_color], [1, self.sphere_color]],
            showscale=False,
            name='Bloch Sphere',
            hovertemplate='<b>Bloch Sphere</b><extra></extra>'
        )
    
    def _add_coordinate_axes(self, fig: go.Figure) -> None:
        """Add X, Y, Z coordinate axes to the figure."""
        axes_data = [
            # X-axis
            {'coords': [[-self.axis_length, self.axis_length], [0, 0], [0, 0]], 'color': 'red', 'name': 'X-axis'},
            # Y-axis
            {'coords': [[0, 0], [-self.axis_length, self.axis_length], [0, 0]], 'color': 'green', 'name': 'Y-axis'},
            # Z-axis
            {'coords': [[0, 0], [0, 0], [-self.axis_length, self.axis_length]], 'color': 'blue', 'name': 'Z-axis'}
        ]
        
        for axis in axes_data:
            fig.add_trace(go.Scatter3d(
                x=axis['coords'][0],
                y=axis['coords'][1],
                z=axis['coords'][2],
                mode='lines',
                line=dict(width=3, color=axis['color']),
                name=axis['name'],
                showlegend=False,
                hovertemplate=f'<b>{axis["name"]}</b><extra></extra>'
            ))
    
    def _add_equator(self, fig: go.Figure) -> None:
        """Add equatorial circle to the figure."""
        theta_eq = np.linspace(0, 2*np.pi, 100)
        x_eq = np.cos(theta_eq)
        y_eq = np.sin(theta_eq)
        z_eq = np.zeros_like(theta_eq)
        
        fig.add_trace(go.Scatter3d(
            x=x_eq, y=y_eq, z=z_eq,
            mode='lines',
            line=dict(width=2, color=self.equator_color, dash='dash'),
            name='Equator',
            showlegend=False,
            hovertemplate='<b>Equator</b><extra></extra>'
        ))
    
    def _add_meridians(self, fig: go.Figure) -> None:
        """Add meridian lines to the figure."""
        phi_values = [0, np.pi/2, np.pi, 3*np.pi/2]
        
        for phi_val in phi_values:
            theta_mer = np.linspace(0, np.pi, 50)
            x_mer = np.sin(theta_mer) * np.cos(phi_val)
            y_mer = np.sin(theta_mer) * np.sin(phi_val)
            z_mer = np.cos(theta_mer)
            
            fig.add_trace(go.Scatter3d(
                x=x_mer, y=y_mer, z=z_mer,
                mode='lines',
                line=dict(width=1, color=self.meridian_color, dash='dot'),
                name=f'Meridian {phi_val:.1f}',
                showlegend=False,
                hovertemplate=f'<b>Meridian φ={phi_val:.2f}</b><extra></extra>'
            ))
    
    def _add_state_vector(self, fig: go.Figure, theta: float, phi: float,
                         color: str = None, name: str = "State Vector") -> None:
        """Add quantum state vector to the figure."""
        if color is None:
            color = self.vector_color
            
        # Convert to Cartesian coordinates
        x, y, z = self._spherical_to_cartesian(theta, phi)
        
        # Add vector arrow
        fig.add_trace(go.Scatter3d(
            x=[0, x],
            y=[0, y],
            z=[0, z],
            mode='lines+markers',
            line=dict(width=6, color=color),
            marker=dict(size=[0, 10], color=color, symbol=['circle', 'diamond']),
            name=name,
            hovertemplate=f'<b>{name}</b><br>' +
                         f'θ: {theta:.3f} rad<br>' +
                         f'φ: {phi:.3f} rad<br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<extra></extra>'
        ))
    
    def _add_axis_labels(self, fig: go.Figure) -> None:
        """Add axis labels to the figure."""
        label_positions = [
            (self.axis_length + 0.1, 0, 0, self.axis_labels['x_pos']),
            (-self.axis_length - 0.1, 0, 0, self.axis_labels['x_neg']),
            (0, self.axis_length + 0.1, 0, self.axis_labels['y_pos']),
            (0, -self.axis_length - 0.1, 0, self.axis_labels['y_neg']),
            (0, 0, self.axis_length + 0.1, self.axis_labels['z_pos']),
            (0, 0, -self.axis_length - 0.1, self.axis_labels['z_neg'])
        ]
        
        for x, y, z, label in label_positions:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='text',
                text=[label],
                textfont=dict(size=14, color='black'),
                showlegend=False,
                hovertemplate=f'<b>Axis Label: {label}</b><extra></extra>'
            ))
    
    def _add_measurement_basis(self, fig: go.Figure, basis: str) -> None:
        """Add measurement basis visualization."""
        if basis.upper() == 'X':
            # Highlight X-axis eigenstates
            positions = [(1, 0, 0), (-1, 0, 0)]
            labels = ['+X', '-X']
            color = 'orange'
        elif basis.upper() == 'Y':
            # Highlight Y-axis eigenstates
            positions = [(0, 1, 0), (0, -1, 0)]
            labels = ['+Y', '-Y']
            color = 'purple'
        else:  # Z basis
            # Highlight Z-axis eigenstates
            positions = [(0, 0, 1), (0, 0, -1)]
            labels = ['|0⟩', '|1⟩']
            color = 'cyan'
        
        for (x, y, z), label in zip(positions, labels):
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=12, color=color, symbol='square'),
                name=f'{basis} basis: {label}',
                hovertemplate=f'<b>{basis} Basis Eigenstate</b><br>' +
                             f'State: {label}<br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>'
            ))
    
    def _add_measurement_projections(self, fig: go.Figure, theta: float, phi: float,
                                   measurement_basis: str) -> None:
        """Add measurement projection visualization."""
        # Calculate state vector in Cartesian coordinates
        x, y, z = self._spherical_to_cartesian(theta, phi)
        
        # Calculate projections based on measurement basis
        if measurement_basis.upper() == 'X':
            projection = x
            prob_plus = (1 + projection) / 2
            prob_minus = (1 - projection) / 2
            basis_name = 'X'
        elif measurement_basis.upper() == 'Y':
            projection = y
            prob_plus = (1 + projection) / 2
            prob_minus = (1 - projection) / 2
            basis_name = 'Y'
        else:  # Z basis
            projection = z
            prob_plus = (1 + projection) / 2
            prob_minus = (1 - projection) / 2
            basis_name = 'Z'
        
        # Add probability annotations
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"<b>{basis_name} Measurement Probabilities:</b><br>" +
                 f"P(+{basis_name}) = {prob_plus:.3f}<br>" +
                 f"P(-{basis_name}) = {prob_minus:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        )
    
    def _spherical_to_cartesian(self, theta: float, phi: float) -> Tuple[float, float, float]:
        """Convert spherical coordinates to Cartesian coordinates."""
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z
    
    def _configure_layout(self, fig: go.Figure, title: str) -> None:
        """Configure the 3D plot layout."""
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title="X",
                    range=[-1.3, 1.3],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="Y",
                    range=[-1.3, 1.3],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                zaxis=dict(
                    title="Z",
                    range=[-1.3, 1.3],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor="white"
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600
        )
    
    def calculate_fidelity(self, theta1: float, phi1: float, 
                          theta2: float, phi2: float) -> float:
        """
        Calculate fidelity between two quantum states on the Bloch sphere.
        
        Args:
            theta1, phi1: Spherical coordinates of first state
            theta2, phi2: Spherical coordinates of second state
            
        Returns:
            Fidelity between the two states (0 to 1)
        """
        # Convert to Cartesian coordinates
        x1, y1, z1 = self._spherical_to_cartesian(theta1, phi1)
        x2, y2, z2 = self._spherical_to_cartesian(theta2, phi2)
        
        # Calculate dot product (overlap)
        dot_product = x1*x2 + y1*y2 + z1*z2
        
        # Fidelity is |⟨ψ1|ψ2⟩|² = (1 + dot_product)/2 for pure states
        fidelity = (1 + dot_product) / 2
        
        return max(0, min(1, fidelity))  # Ensure bounds [0, 1]
    
    def generate_random_state(self) -> Tuple[float, float]:
        """
        Generate random quantum state on the Bloch sphere.
        
        Returns:
            Tuple of (theta, phi) for random state
        """
        # Uniform distribution on sphere surface
        theta = np.arccos(1 - 2 * np.random.random())  # Uniform in cos(theta)
        phi = 2 * np.pi * np.random.random()
        
        return theta, phi
    
    def get_pauli_expectation_values(self, theta: float, phi: float) -> Dict[str, float]:
        """
        Calculate Pauli expectation values for a quantum state.
        
        Args:
            theta: Polar angle
            phi: Azimuthal angle
            
        Returns:
            Dictionary with expectation values for X, Y, Z
        """
        x, y, z = self._spherical_to_cartesian(theta, phi)
        
        return {
            'X': x,
            'Y': y,
            'Z': z
        }
