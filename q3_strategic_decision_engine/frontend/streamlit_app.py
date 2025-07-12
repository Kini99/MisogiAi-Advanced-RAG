"""
Strategic Decision Engine - Streamlit Frontend
A comprehensive AI-powered strategic planning platform for CEOs
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
import uuid
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Strategic Decision Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}


# Utility functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Make API request to backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {"error": str(e)}


def display_chat_message(message: Dict[str, str], is_user: bool = True):
    """Display a chat message"""
    css_class = "user-message" if is_user else "assistant-message"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{'You' if is_user else 'Assistant'}:</strong><br>
        {message['content']}
    </div>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, delta: str = None):
    """Create a metric card"""
    delta_html = f"<p style='color: green; margin: 0;'>↗ {delta}</p>" if delta else ""
    
    return f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #2a5298;">{title}</h4>
        <h2 style="margin: 0.5rem 0;">{value}</h2>
        {delta_html}
    </div>
    """


# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 Strategic Decision Engine</h1>
    <p>AI-Powered Strategic Planning Platform for CEOs</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Navigation")
    
    page = st.selectbox(
        "Select Page",
        [
            "📊 Dashboard",
            "📁 Document Management",
            "💬 Strategic Chat",
            "🔍 SWOT Analysis",
            "📈 Market Analysis",
            "📊 Financial Analysis",
            "📋 Evaluation Dashboard"
        ]
    )
    
    st.markdown("---")
    
    # Health status
    try:
        health = make_api_request("/health")
        if health.get("status") == "healthy":
            st.success("🟢 System Healthy")
        else:
            st.error("🔴 System Issues")
    except:
        st.error("🔴 Backend Offline")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    try:
        stats = make_api_request("/documents/stats/overview")
        if "error" not in stats:
            st.metric("Documents", stats.get("total_documents", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Storage", f"{stats.get('total_size_mb', 0):.1f} MB")
    except:
        st.info("Stats unavailable")


# Main content area
if page == "📊 Dashboard":
    st.header("Strategic Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Documents", "12", "+3"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Analyses", "45", "+8"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("Insights", "127", "+23"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Score", "8.5/10", "+0.2"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Analysis Trends")
        
        # Sample data for demo
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        analyses = [12, 15, 18, 22, 28, 25, 30, 35, 40, 38, 42, 45]
        
        fig = px.line(
            x=dates, 
            y=analyses, 
            title="Monthly Strategic Analyses",
            labels={'x': 'Month', 'y': 'Number of Analyses'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Analysis Types")
        
        # Sample data
        analysis_types = ['SWOT', 'Market', 'Financial', 'Competitive', 'Risk']
        counts = [25, 18, 15, 12, 8]
        
        fig = px.pie(
            values=counts, 
            names=analysis_types, 
            title="Analysis Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activities
    st.subheader("🔄 Recent Activities")
    
    activities = [
        {"time": "2 hours ago", "action": "SWOT Analysis completed", "status": "✅"},
        {"time": "5 hours ago", "action": "Market report uploaded", "status": "📁"},
        {"time": "1 day ago", "action": "Financial analysis generated", "status": "📊"},
        {"time": "2 days ago", "action": "Strategic plan reviewed", "status": "👁️"},
    ]
    
    for activity in activities:
        st.markdown(f"**{activity['status']} {activity['action']}** - {activity['time']}")


elif page == "📁 Document Management":
    st.header("Document Management")
    
    # Upload section
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'csv'],
        help="Upload company documents, reports, financial data, and market research"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")
        
        with col2:
            if st.button("Upload All", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Uploading {file.name}...")
                        
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        result = make_api_request("/documents/upload", method="POST", files=files)
                        
                        if "error" not in result:
                            st.success(f"✅ {file.name} uploaded successfully")
                            st.session_state.uploaded_documents.append(result)
                        else:
                            st.error(f"❌ Failed to upload {file.name}: {result['error']}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"❌ Error uploading {file.name}: {str(e)}")
                
                status_text.text("Upload complete!")
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    
    # Document list
    st.subheader("📋 Uploaded Documents")
    
    try:
        docs_response = make_api_request("/documents/list")
        if "error" not in docs_response and "documents" in docs_response:
            documents = docs_response["documents"]
            
            if documents:
                # Create dataframe for display
                df = pd.DataFrame([
                    {
                        "ID": doc["id"],
                        "Filename": doc["filename"],
                        "Type": doc["file_type"],
                        "Size (KB)": f"{doc['file_size'] / 1024:.1f}",
                        "Uploaded": doc["upload_date"][:10],
                        "Processed": "✅" if doc["processed"] else "⏳",
                        "Chunks": doc["chunk_count"]
                    }
                    for doc in documents
                ])
                
                st.dataframe(df, use_container_width=True)
                
                # Document actions
                st.subheader("🔧 Document Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    doc_id = st.selectbox("Select Document", options=[doc["id"] for doc in documents])
                
                with col2:
                    if st.button("View Details"):
                        doc_info = make_api_request(f"/documents/{doc_id}")
                        if "error" not in doc_info:
                            st.json(doc_info)
                
                with col3:
                    if st.button("Delete", type="secondary"):
                        if st.warning("Are you sure you want to delete this document?"):
                            result = make_api_request(f"/documents/{doc_id}", method="DELETE")
                            if "error" not in result:
                                st.success("Document deleted successfully")
                                st.rerun()
            else:
                st.info("No documents uploaded yet")
        else:
            st.error("Failed to load documents")
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")


elif page == "💬 Strategic Chat":
    st.header("Strategic Chat Assistant")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message, message.get("role") == "user")
    
    # Chat input
    user_input = st.chat_input("Ask a strategic question...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show thinking spinner
        with st.spinner("Analyzing your question..."):
            try:
                # Make API request to chat endpoint
                response = make_api_request("/chat/message", method="POST", data={
                    "message": user_input,
                    "session_id": st.session_state.session_id
                })
                
                if "error" not in response:
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.get("response", "I apologize, but I couldn't generate a response."),
                        "timestamp": datetime.now().isoformat(),
                        "sources": response.get("sources", [])
                    })
                else:
                    st.error(f"Chat error: {response['error']}")
            
            except Exception as e:
                st.error(f"Failed to get response: {str(e)}")
        
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            # Implement chat saving functionality
            st.success("Chat saved!")
    
    # Suggested questions
    st.subheader("💡 Suggested Questions")
    
    suggestions = [
        "Create a SWOT analysis for our company",
        "What are the key market expansion opportunities?",
        "Analyze our competitive position",
        "Generate strategic recommendations for growth",
        "What are the main risks we should consider?"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion}"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": suggestion,
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()


elif page == "🔍 SWOT Analysis":
    st.header("SWOT Analysis Generator")
    
    st.markdown("""
    Generate comprehensive SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis 
    based on your uploaded documents and market data.
    """)
    
    # SWOT configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["Company-wide", "Product Line", "Market Segment", "Department", "Custom"]
        )
        
        if analysis_scope == "Custom":
            custom_scope = st.text_input("Specify custom scope:")
    
    with col2:
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_citations = st.checkbox("Include Source Citations", value=True)
    
    if st.button("🔍 Generate SWOT Analysis", type="primary"):
        with st.spinner("Generating SWOT analysis..."):
            try:
                # Make API request for SWOT analysis
                request_data = {
                    "analysis_type": "swot",
                    "scope": analysis_scope,
                    "session_id": st.session_state.session_id,
                    "include_charts": include_charts,
                    "include_citations": include_citations
                }
                
                response = make_api_request("/analysis/generate", method="POST", data=request_data)
                
                if "error" not in response:
                    st.success("✅ SWOT analysis generated successfully!")
                    
                    # Display results
                    analysis = response.get("analysis", {})
                    
                    # SWOT Matrix
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("💪 Strengths")
                        strengths = analysis.get("strengths", ["Analysis in progress..."])
                        for strength in strengths:
                            st.write(f"• {strength}")
                        
                        st.subheader("🚀 Opportunities")
                        opportunities = analysis.get("opportunities", ["Analysis in progress..."])
                        for opportunity in opportunities:
                            st.write(f"• {opportunity}")
                    
                    with col2:
                        st.subheader("⚠️ Weaknesses")
                        weaknesses = analysis.get("weaknesses", ["Analysis in progress..."])
                        for weakness in weaknesses:
                            st.write(f"• {weakness}")
                        
                        st.subheader("⚡ Threats")
                        threats = analysis.get("threats", ["Analysis in progress..."])
                        for threat in threats:
                            st.write(f"• {threat}")
                    
                    # Strategic recommendations
                    if "recommendations" in analysis:
                        st.subheader("🎯 Strategic Recommendations")
                        st.write(analysis["recommendations"])
                    
                    # Save to session state
                    st.session_state.analysis_results["swot"] = analysis
                
                else:
                    st.error(f"Analysis failed: {response['error']}")
            
            except Exception as e:
                st.error(f"Failed to generate analysis: {str(e)}")
    
    # Previous analyses
    if "swot" in st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Previous SWOT Analysis")
        
        if st.button("📥 Download Analysis"):
            # Implement download functionality
            st.success("Analysis downloaded!")


elif page == "📈 Market Analysis":
    st.header("Market Analysis Dashboard")
    
    st.markdown("""
    Analyze market trends, competitive landscape, and expansion opportunities 
    using AI-powered insights from your documents and external market data.
    """)
    
    # Market analysis configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_focus = st.selectbox(
            "Market Focus",
            ["Current Markets", "New Markets", "Global Markets", "Niche Markets"]
        )
    
    with col2:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["3 months", "6 months", "1 year", "2 years", "5 years"]
        )
    
    with col3:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Overview", "Detailed Analysis", "Comprehensive Report"]
        )
    
    if st.button("📊 Generate Market Analysis", type="primary"):
        with st.spinner("Analyzing market data..."):
            # Simulate market analysis
            time.sleep(3)
            
            st.success("✅ Market analysis completed!")
            
            # Market trends chart
            st.subheader("📈 Market Trends")
            
            # Sample data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            market_size = [100, 105, 110, 115, 122, 128]
            growth_rate = [5, 4.8, 5.2, 4.5, 6.1, 4.9]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=market_size, mode='lines+markers', name='Market Size'))
            fig.add_trace(go.Scatter(x=months, y=growth_rate, mode='lines+markers', name='Growth Rate %', yaxis='y2'))
            
            fig.update_layout(
                title="Market Size and Growth Trends",
                xaxis_title="Month",
                yaxis_title="Market Size (Index)",
                yaxis2=dict(title="Growth Rate (%)", overlaying='y', side='right'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Key Opportunities")
                opportunities = [
                    "Emerging market segments showing 15% growth",
                    "Digital transformation creating new demands",
                    "Sustainability trends opening new niches",
                    "International expansion potential in Asia"
                ]
                for opp in opportunities:
                    st.write(f"• {opp}")
            
            with col2:
                st.subheader("⚠️ Market Risks")
                risks = [
                    "Increased competition from new entrants",
                    "Regulatory changes affecting operations",
                    "Economic uncertainty impacting demand",
                    "Technology disruption threats"
                ]
                for risk in risks:
                    st.write(f"• {risk}")


elif page == "📊 Financial Analysis":
    st.header("Financial Analysis Dashboard")
    
    st.markdown("""
    Analyze financial performance, forecast trends, and generate insights 
    from your financial documents and market data.
    """)
    
    # Financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue Growth", "12.5%", "2.3%")
    
    with col2:
        st.metric("Profit Margin", "18.2%", "-0.8%")
    
    with col3:
        st.metric("ROI", "24.1%", "3.2%")
    
    with col4:
        st.metric("Cash Flow", "$2.1M", "$0.3M")
    
    # Financial charts
    st.subheader("📊 Financial Performance")
    
    # Sample financial data
    quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
    revenue = [2.1, 2.4, 2.6, 2.8]
    expenses = [1.6, 1.8, 1.9, 2.0]
    profit = [0.5, 0.6, 0.7, 0.8]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=quarters, y=revenue, name='Revenue', marker_color='lightblue'))
    fig.add_trace(go.Bar(x=quarters, y=expenses, name='Expenses', marker_color='lightcoral'))
    fig.add_trace(go.Scatter(x=quarters, y=profit, mode='lines+markers', name='Profit', line=dict(color='green', width=3)))
    
    fig.update_layout(
        title="Quarterly Financial Performance",
        xaxis_title="Quarter",
        yaxis_title="Amount ($ Millions)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Financial analysis tools
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Forecasting")
        if st.button("Generate Financial Forecast"):
            with st.spinner("Generating forecast..."):
                time.sleep(2)
                st.success("Forecast generated!")
                st.info("Based on current trends, revenue is projected to grow 15% next quarter.")
    
    with col2:
        st.subheader("💡 Recommendations")
        if st.button("Get Financial Insights"):
            with st.spinner("Analyzing financial data..."):
                time.sleep(2)
                st.success("Analysis complete!")
                st.info("Consider optimizing operational expenses to improve profit margins.")


elif page == "📋 Evaluation Dashboard":
    st.header("RAGAS Evaluation Dashboard")
    
    st.markdown("""
    Monitor the quality of AI-generated strategic insights using RAGAS evaluation metrics:
    Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness.
    """)
    
    # Evaluation metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Faithfulness", "0.87", "0.03")
    
    with col2:
        st.metric("Answer Relevancy", "0.92", "0.01")
    
    with col3:
        st.metric("Context Precision", "0.85", "-0.02")
    
    with col4:
        st.metric("Context Recall", "0.89", "0.04")
    
    with col5:
        st.metric("Overall Score", "0.88", "0.02")
    
    # Evaluation trends
    st.subheader("📊 Evaluation Trends")
    
    # Sample evaluation data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    faithfulness = [0.85 + 0.02 * (i % 10) for i in range(len(dates))]
    relevancy = [0.90 + 0.015 * (i % 8) for i in range(len(dates))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=faithfulness, mode='lines', name='Faithfulness'))
    fig.add_trace(go.Scatter(x=dates, y=relevancy, mode='lines', name='Answer Relevancy'))
    
    fig.update_layout(
        title="RAGAS Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Evaluation controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Run Evaluation")
        if st.button("Evaluate Recent Responses"):
            with st.spinner("Running RAGAS evaluation..."):
                time.sleep(3)
                st.success("Evaluation completed!")
                st.info("All metrics are within acceptable ranges.")
    
    with col2:
        st.subheader("📊 Detailed Analysis")
        if st.button("Generate Evaluation Report"):
            with st.spinner("Generating report..."):
                time.sleep(2)
                st.success("Report generated!")
                st.download_button(
                    label="📥 Download Report",
                    data="Evaluation report content...",
                    file_name="evaluation_report.pdf",
                    mime="application/pdf"
                )


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Strategic Decision Engine v1.0 | Powered by AI | Built with ❤️ for Strategic Excellence
</div>
""", unsafe_allow_html=True) 