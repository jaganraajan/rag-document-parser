"""
Streamlit UI for Hybrid Document Normalization Pipeline.

This module provides a web interface for multi-file upload, status display,
normalized JSON preview/download, interpretation log viewing, and cost summaries.
"""

import streamlit as st
import os
import sys
import json
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalization.pipeline.document_pipeline import DocumentPipeline
from src.normalization.storage.repository import DocumentRepository
from src.normalization.logs.interpretation import InterpretationLogger


# Page configuration
st.set_page_config(
    page_title="Hybrid Document Normalization Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'repository' not in st.session_state:
    st.session_state.repository = DocumentRepository()
if 'logger' not in st.session_state:
    st.session_state.logger = InterpretationLogger()


def initialize_pipeline():
    """Initialize the document pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("Initializing pipeline..."):
            st.session_state.pipeline = DocumentPipeline()
        st.success("Pipeline initialized successfully!")
        return st.session_state.pipeline
    return st.session_state.pipeline


def main():
    """Main Streamlit application."""
    st.title("📄 Hybrid Document Normalization Pipeline")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Document Upload", "Document Library", "Pipeline Stats", "Interpretation Logs", "Settings"]
    )
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    if page == "Document Upload":
        document_upload_page(pipeline)
    elif page == "Document Library":
        document_library_page()
    elif page == "Pipeline Stats":
        pipeline_stats_page(pipeline)
    elif page == "Interpretation Logs":
        interpretation_logs_page()
    elif page == "Settings":
        settings_page()


def document_upload_page(pipeline: DocumentPipeline):
    """Document upload and processing page."""
    st.header("📤 Document Upload & Processing")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to process",
        type=['pdf', 'png', 'jpg', 'jpeg', 'eml', 'txt', 'html'],
        accept_multiple_files=True,
        help="Supported formats: PDF, images (PNG, JPG), email (.eml), text files"
    )
    
    # Processing settings
    col1, col2 = st.columns(2)
    
    with col1:
        max_cost_per_doc = st.number_input(
            "Max LLM cost per document ($)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Maximum cost limit for LLM calls per document"
        )
    
    with col2:
        llm_provider = st.selectbox(
            "LLM Provider",
            ["dummy", "openai", "gemini"],
            help="Select the LLM provider for field extraction"
        )
    
    # Process documents
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        process_documents(pipeline, uploaded_files, max_cost_per_doc, llm_provider)
    
    # Display processing results
    if st.session_state.processed_docs:
        st.subheader("📊 Processing Results")
        display_processing_results()


def process_documents(pipeline: DocumentPipeline, uploaded_files, max_cost_per_doc: float, llm_provider: str):
    """Process uploaded documents."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                document, log_file = pipeline.process_document(tmp_file_path, max_cost_per_doc)
                
                result = {
                    'filename': uploaded_file.name,
                    'doc_id': document.doc_id,
                    'status': 'success',
                    'signature_id': document.processing_meta.signature_id,
                    'signature_match': document.processing_meta.signature_match_score,
                    'fields_extracted': len(document.key_values),
                    'cost_usd': document.processing_meta.total_cost_usd,
                    'processing_time': document.ingest_metadata.processing_time_seconds,
                    'coverage_stats': document.processing_meta.coverage_stats,
                    'log_file': log_file,
                    'document': document
                }
                
                st.success(f"✅ Processed {uploaded_file.name}")
                
            except Exception as e:
                result = {
                    'filename': uploaded_file.name,
                    'doc_id': None,
                    'status': 'error',
                    'error': str(e)
                }
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
            
            results.append(result)
            
        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
    
    # Update session state
    st.session_state.processed_docs.extend(results)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")


def display_processing_results():
    """Display processing results in a table."""
    # Create DataFrame
    data = []
    for result in st.session_state.processed_docs:
        if result['status'] == 'success':
            coverage = result.get('coverage_stats', {})
            data.append({
                'File': result['filename'],
                'Doc ID': result['doc_id'],
                'Signature': result['signature_id'][:8] + '...' if result['signature_id'] else 'N/A',
                'Match Score': f"{result['signature_match']:.2f}",
                'Fields': result['fields_extracted'],
                'Rule Coverage': f"{coverage.get('rule_coverage', 0):.1%}",
                'Cost ($)': f"{result['cost_usd']:.3f}",
                'Time (s)': f"{result['processing_time']:.1f}",
                'Status': '✅ Success'
            })
        else:
            data.append({
                'File': result['filename'],
                'Doc ID': 'N/A',
                'Signature': 'N/A',
                'Match Score': 'N/A',
                'Fields': 'N/A',
                'Rule Coverage': 'N/A',
                'Cost ($)': 'N/A',
                'Time (s)': 'N/A',
                'Status': f"❌ {result.get('error', 'Error')}"
            })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Document detail section
        st.subheader("📋 Document Details")
        
        # Select document for details
        doc_options = [f"{r['filename']} ({r['doc_id']})" for r in st.session_state.processed_docs if r['status'] == 'success']
        
        if doc_options:
            selected_doc = st.selectbox("Select document for details:", doc_options)
            
            if selected_doc:
                # Find selected document
                doc_id = selected_doc.split('(')[-1].rstrip(')')
                selected_result = next(r for r in st.session_state.processed_docs if r.get('doc_id') == doc_id)
                
                display_document_details(selected_result)


def display_document_details(result: Dict[str, Any]):
    """Display detailed information for a selected document."""
    document = result['document']
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 JSON Preview", "📊 Key-Values", "📋 Sections", "🔍 Logs"])
    
    with tab1:
        st.subheader("Normalized JSON")
        
        # Download button
        json_str = json.dumps(document.to_dict(), indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{document.doc_id}_normalized.json",
            mime="application/json"
        )
        
        # Display JSON
        st.json(document.to_dict())
    
    with tab2:
        st.subheader("Extracted Key-Value Pairs")
        
        if document.key_values:
            kv_data = []
            for kv in document.key_values:
                kv_data.append({
                    'Field': kv.key,
                    'Value': str(kv.value),
                    'Confidence': f"{kv.confidence:.2f}",
                    'Method': kv.extraction_method.title()
                })
            
            kv_df = pd.DataFrame(kv_data)
            st.dataframe(kv_df, use_container_width=True)
        else:
            st.info("No key-value pairs extracted")
    
    with tab3:
        st.subheader("Document Sections")
        
        if document.sections:
            for i, section in enumerate(document.sections):
                with st.expander(f"Section {i+1}: {section.title}"):
                    st.write(section.content)
        else:
            st.info("No sections found")
    
    with tab4:
        st.subheader("Interpretation Logs")
        
        if result.get('log_file'):
            display_log_events(result['log_file'])


def document_library_page():
    """Document library and search page."""
    st.header("📚 Document Library")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file_type_filter = st.selectbox(
            "File Type",
            ["All", "pdf", "image", "email", "text", "html"]
        )
    
    with col2:
        min_coverage = st.slider(
            "Min Rule Coverage",
            0.0, 1.0, 0.0, 0.1
        )
    
    with col3:
        max_cost = st.number_input(
            "Max Cost ($)",
            0.0, 10.0, 10.0
        )
    
    # Build filters
    filters = {}
    if file_type_filter != "All":
        filters["file_type"] = file_type_filter
    if min_coverage > 0:
        filters["min_coverage"] = min_coverage
    if max_cost < 10.0:
        filters["max_cost"] = max_cost
    
    # Get documents
    documents = st.session_state.repository.search_documents(**filters)
    
    if documents:
        # Display documents table
        data = []
        for doc in documents:
            coverage = doc.get('coverage_stats', {})
            data.append({
                'File': doc['filename'],
                'Doc ID': doc['doc_id'],
                'Type': doc['file_type'],
                'Uploaded': doc['uploaded_at'][:10],
                'Signature': doc.get('signature_id', 'N/A')[:8] + '...' if doc.get('signature_id') else 'N/A',
                'Fields': doc['key_values_count'],
                'Coverage': f"{coverage.get('rule_coverage', 0):.1%}",
                'Cost': f"${doc['total_cost_usd']:.3f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Document selection for viewing
        if st.selectbox("Select document to view:", ["None"] + [doc['doc_id'] for doc in documents]) != "None":
            selected_doc_id = st.selectbox("Select document to view:", ["None"] + [doc['doc_id'] for doc in documents])
            if selected_doc_id != "None":
                document = st.session_state.repository.get_document(selected_doc_id)
                if document:
                    st.json(document.to_dict())
    else:
        st.info("No documents found matching the criteria")


def pipeline_stats_page(pipeline: DocumentPipeline):
    """Pipeline statistics page."""
    st.header("📈 Pipeline Statistics")
    
    try:
        stats = pipeline.get_pipeline_stats()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                stats['repository']['total_documents']
            )
        
        with col2:
            st.metric(
                "Total Signatures",
                stats['signatures']['total_signatures']
            )
        
        with col3:
            st.metric(
                "Total Rules",
                stats['rules']['total_rules']
            )
        
        with col4:
            st.metric(
                "Total LLM Calls",
                stats['llm']['total_calls']
            )
        
        # Detailed stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Repository Stats")
            repo_stats = stats['repository']
            st.json(repo_stats)
        
        with col2:
            st.subheader("🤖 LLM Stats")
            llm_stats = stats['llm']
            st.json(llm_stats)
        
        # Cost summary
        st.subheader("💰 Cost Summary")
        cost_stats = stats['costs']
        st.json(cost_stats)
        
    except Exception as e:
        st.error(f"Error loading stats: {e}")


def interpretation_logs_page():
    """Interpretation logs viewing page."""
    st.header("🔍 Interpretation Logs")
    
    # Get recent logs
    recent_logs = st.session_state.logger.get_recent_logs(limit=100)
    
    if recent_logs:
        # Create DataFrame
        log_data = []
        for log in recent_logs:
            log_data.append({
                'Timestamp': log.timestamp[:19],
                'Doc ID': log.doc_id,
                'Stage': log.stage,
                'Action': log.action,
                'Decision': log.decision,
                'Cost ($)': f"{log.cost_usd:.4f}" if log.cost_usd > 0 else "-",
                'Detail': log.detail[:100] + "..." if len(log.detail) > 100 else log.detail
            })
        
        logs_df = pd.DataFrame(log_data)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            stage_filter = st.selectbox("Filter by Stage:", ["All"] + list(logs_df['Stage'].unique()))
        with col2:
            doc_filter = st.selectbox("Filter by Document:", ["All"] + list(logs_df['Doc ID'].unique()))
        
        # Apply filters
        filtered_df = logs_df.copy()
        if stage_filter != "All":
            filtered_df = filtered_df[filtered_df['Stage'] == stage_filter]
        if doc_filter != "All":
            filtered_df = filtered_df[filtered_df['Doc ID'] == doc_filter]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export logs
        if st.button("📥 Export Logs to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"interpretation_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No interpretation logs found")


def display_log_events(log_file_path: str):
    """Display log events from a specific log file."""
    try:
        events = st.session_state.logger.read_log_file(log_file_path)
        
        if events:
            for event in events:
                with st.expander(f"{event.stage} - {event.action} ({event.timestamp[:19]})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Decision:** {event.decision}")
                        st.write(f"**Detail:** {event.detail}")
                    with col2:
                        st.write(f"**Cost:** ${event.cost_usd:.4f}")
                        if event.metadata:
                            st.write("**Metadata:**")
                            st.json(event.metadata)
        else:
            st.info("No log events found")
    except Exception as e:
        st.error(f"Error reading log file: {e}")


def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings")
    
    # Pipeline configuration
    st.subheader("🔧 Pipeline Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Rules Directory", value="rules", disabled=True)
        st.text_input("Signatures Directory", value="signatures", disabled=True)
    
    with col2:
        st.text_input("Outputs Directory", value="outputs", disabled=True)
        st.text_input("Logs Directory", value="logs", disabled=True)
    
    # Clear caches
    st.subheader("🧹 Cache Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear LLM Cache"):
            if st.session_state.pipeline:
                st.session_state.pipeline.clear_caches()
                st.success("LLM cache cleared!")
    
    with col2:
        if st.button("Clear Session Data"):
            st.session_state.processed_docs = []
            st.success("Session data cleared!")
    
    with col3:
        if st.button("Reset Pipeline"):
            st.session_state.pipeline = None
            st.success("Pipeline reset!")
    
    # Export configuration
    st.subheader("📤 Export Configuration")
    
    if st.button("Export Pipeline Config"):
        if st.session_state.pipeline:
            config_data = st.session_state.pipeline.get_pipeline_stats()
            config_json = json.dumps(config_data, indent=2, default=str)
            st.download_button(
                label="Download Configuration",
                data=config_json,
                file_name=f"pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()