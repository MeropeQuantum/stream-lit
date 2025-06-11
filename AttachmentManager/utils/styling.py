"""
Enterprise-grade styling utilities for the quantum control dashboard.
Provides consistent styling, colors, and UI components.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class EnterpriseTheme:
    """Enterprise color scheme and styling constants."""
    
    # Color palette
    PRIMARY = "#00d4aa"
    SECONDARY = "#1e2139"
    ACCENT = "#4f46e5"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    INFO = "#3b82f6"
    
    # Background colors
    BG_PRIMARY = "#0a0e1a"
    BG_SECONDARY = "#1e2139"
    BG_CARD = "#2d3748"
    BG_ACCENT = "#374151"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#d1d5db"
    TEXT_MUTED = "#9ca3af"
    
    # Status colors
    STATUS_OPERATIONAL = "#10b981"
    STATUS_WARNING = "#f59e0b"
    STATUS_CRITICAL = "#ef4444"
    STATUS_MAINTENANCE = "#6366f1"
    
    # Chart colors
    CHART_COLORS = [
        "#00d4aa", "#4f46e5", "#f59e0b", "#ef4444", 
        "#10b981", "#8b5cf6", "#06b6d4", "#f97316"
    ]

def apply_enterprise_style():
    """Apply enterprise-grade CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0e1a 0%, #1e2139 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e2139 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4b5563;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.2);
        border-color: #00d4aa;
    }
    
    /* Status indicators */
    .status-operational {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-critical {
        color: #ef4444;
        font-weight: 600;
    }
    
    .status-maintenance {
        color: #6366f1;
        font-weight: 600;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #10b981 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e2139 0%, #2d3748 100%);
        border-right: 1px solid #374151;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #2d3748 0%, #374151 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4b5563;
        text-align: center;
    }
    
    /* Alert styling */
    .alert-success {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #92400e 0%, #f59e0b 100%);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-error {
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #374151;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: #374151;
    }
    
    .stProgress .st-bp {
        background: linear-gradient(90deg, #00d4aa 0%, #10b981 100%);
    }
    
    /* Custom classes for enterprise components */
    .enterprise-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .enterprise-subtitle {
        font-size: 1.125rem;
        color: #d1d5db;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .system-status-good {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #065f46;
        color: #10b981;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .system-status-warning {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #92400e;
        color: #f59e0b;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .system-status-critical {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #991b1b;
        color: #ef4444;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e2139;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .5; }
    }
    </style>
    """, unsafe_allow_html=True)

def create_enterprise_metric_card(title: str, value: str, delta: str = "", delta_color: str = "normal"):
    """Create an enterprise-styled metric card."""
    delta_class = ""
    if delta and delta != "":
        if delta_color == "normal":
            delta_class = "status-operational" if delta.startswith('+') else "status-critical"
        elif delta_color == "inverse":
            delta_class = "status-critical" if delta.startswith('+') else "status-operational"
    
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta and delta != "" else ""
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 0.875rem; color: #9ca3af; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.25rem;">{value}</div>
        {delta_html}
    </div>
    """

def create_status_badge(status: str, text: str = ""):
    """Create a status badge with enterprise styling."""
    if text == "":
        text = status
    
    status_classes = {
        "operational": "system-status-good",
        "warning": "system-status-warning", 
        "critical": "system-status-critical",
        "good": "system-status-good",
        "maintenance": "system-status-warning"
    }
    
    css_class = status_classes.get(status.lower(), "system-status-good")
    return f'<span class="{css_class}">{text}</span>'

def get_enterprise_plotly_theme():
    """Get enterprise Plotly theme configuration."""
    return {
        'layout': {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'family': 'Inter, sans-serif',
                'color': '#ffffff',
                'size': 12
            },
            'colorway': EnterpriseTheme.CHART_COLORS,
            'xaxis': {
                'gridcolor': '#374151',
                'linecolor': '#4b5563',
                'tickcolor': '#6b7280',
                'zerolinecolor': '#4b5563'
            },
            'yaxis': {
                'gridcolor': '#374151', 
                'linecolor': '#4b5563',
                'tickcolor': '#6b7280',
                'zerolinecolor': '#4b5563'
            },
            'legend': {
                'bgcolor': 'rgba(45, 55, 72, 0.8)',
                'bordercolor': '#4b5563',
                'borderwidth': 1
            }
        }
    }

def create_enterprise_header(title: str, subtitle: str = "", status: str = ""):
    """Create an enterprise-styled header."""
    status_html = ""
    if status and status != "":
        status_html = f'<div style="margin-top: 1rem;">{create_status_badge(status)}</div>'
    
    subtitle_html = ""
    if subtitle and subtitle != "":
        subtitle_html = f'<div class="enterprise-subtitle">{subtitle}</div>'
    
    return f"""
    <div class="main-header fade-in">
        <div class="enterprise-title">{title}</div>
        {subtitle_html}
        {status_html}
    </div>
    """

def create_alert_box(message: str, alert_type: str = "info", icon: str = ""):
    """Create an enterprise-styled alert box."""
    icons = {
        "success": "✓",
        "warning": "⚠",
        "error": "✕",
        "info": "ℹ"
    }
    
    if icon == "":
        icon = icons.get(alert_type, "ℹ")
    
    alert_class = f"alert-{alert_type}"
    
    return f"""
    <div class="{alert_class}">
        <span style="margin-right: 0.5rem; font-weight: 600;">{icon}</span>
        {message}
    </div>
    """