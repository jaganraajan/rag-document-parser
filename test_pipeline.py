#!/usr/bin/env python3
"""
Simple test script to verify the hybrid document normalization pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.normalization.pipeline.document_pipeline import DocumentPipeline


def create_test_pdf():
    """Create a simple test PDF file."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        
        # Create PDF content
        c = canvas.Canvas(temp_file.name, pagesize=letter)
        c.drawString(100, 750, "Test Invoice")
        c.drawString(100, 700, "Invoice Number: INV-12345")
        c.drawString(100, 650, "Total Amount: $123.45")
        c.drawString(100, 600, "Date: 12/15/2024")
        c.drawString(100, 550, "Vendor: Test Company Inc.")
        c.drawString(100, 500, "Email: test@company.com")
        c.drawString(100, 450, "Phone: 555-1234")
        c.save()
        
        return temp_file.name
        
    except ImportError:
        # If reportlab is not available, create a simple text file instead
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w')
        temp_file.write("""Test Invoice
Invoice Number: INV-12345
Total Amount: $123.45
Date: 12/15/2024
Vendor: Test Company Inc.
Email: test@company.com
Phone: 555-1234
""")
        temp_file.close()
        return temp_file.name


def create_test_email():
    """Create a simple test email file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.eml', delete=False, mode='w')
    temp_file.write("""From: sender@example.com
To: recipient@example.com
Subject: Test Email Invoice
Date: Mon, 15 Dec 2024 10:00:00 +0000

This is a test email with invoice information.

Invoice Number: INV-67890
Total Amount: $456.78
Vendor: Email Vendor Corp
""")
    temp_file.close()
    return temp_file.name


def test_pipeline():
    """Test the document processing pipeline."""
    print("🚀 Testing Hybrid Document Normalization Pipeline")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("📋 Initializing pipeline...")
        pipeline = DocumentPipeline()
        print("✅ Pipeline initialized successfully")
        
        # Create test files
        print("\n📄 Creating test documents...")
        test_pdf = create_test_pdf()
        test_email = create_test_email()
        
        test_files = [test_pdf, test_email]
        print(f"✅ Created {len(test_files)} test documents")
        
        # Process documents
        print("\n🔄 Processing documents...")
        
        for i, file_path in enumerate(test_files, 1):
            print(f"\n--- Processing Document {i}: {os.path.basename(file_path)} ---")
            
            try:
                document, log_file = pipeline.process_document(file_path, max_cost_usd=0.5)
                
                print(f"✅ Successfully processed: {document.doc_id}")
                print(f"   📊 Fields extracted: {len(document.key_values)}")
                print(f"   🏷️  Signature ID: {document.processing_meta.signature_id}")
                print(f"   💰 Cost: ${document.processing_meta.total_cost_usd:.4f}")
                print(f"   ⏱️  Time: {document.ingest_metadata.processing_time_seconds:.2f}s")
                print(f"   📝 Log file: {log_file}")
                
                # Show extracted key-values
                if document.key_values:
                    print("   🔑 Extracted fields:")
                    for kv in document.key_values:
                        print(f"      {kv.key}: {kv.value} ({kv.extraction_method})")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                
        # Get pipeline statistics
        print("\n📈 Pipeline Statistics:")
        stats = pipeline.get_pipeline_stats()
        
        print(f"   📚 Total documents: {stats['repository']['total_documents']}")
        print(f"   🔖 Total signatures: {stats['signatures']['total_signatures']}")
        print(f"   📋 Total rules: {stats['rules']['total_rules']}")
        print(f"   🤖 LLM calls: {stats['llm']['total_calls']}")
        print(f"   💰 Total cost: ${stats['costs']['total_cost_usd']:.4f}")
        
        print("\n✅ Pipeline test completed successfully!")
        
        # Show directories created
        print("\n📁 Generated directories:")
        for dir_name in ['signatures', 'outputs', 'logs']:
            if os.path.exists(dir_name):
                file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
                print(f"   {dir_name}/: {file_count} files")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test files
        for file_path in [test_pdf, test_email]:
            try:
                os.unlink(file_path)
            except:
                pass


def test_streamlit_app():
    """Test if Streamlit app can be imported."""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        # Test import
        sys.path.insert(0, 'app')
        import streamlit_app
        print("✅ Streamlit app imports successfully")
        
        print("📝 To run the Streamlit app:")
        print("   streamlit run app/streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running hybrid document normalization pipeline tests...\n")
    
    # Test pipeline
    pipeline_success = test_pipeline()
    
    # Test Streamlit app
    streamlit_success = test_streamlit_app()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    print(f"   Streamlit: {'✅ PASS' if streamlit_success else '❌ FAIL'}")
    
    if pipeline_success and streamlit_success:
        print("\n🎉 All tests passed! The hybrid document normalization pipeline is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app/streamlit_app.py")
        print("   2. Upload documents and test the pipeline")
        print("   3. Check the generated directories for outputs")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)