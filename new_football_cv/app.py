import gradio as gr
import io
import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

#databricks apps deploy app-football-cv --source-code-path /Workspace/Users/ali.azzouz@databricks.com/football_cv/app-football-cv

# Initialize Databricks workspace client
w = WorkspaceClient()

genie_space_id = "01f099ea3cf91bfcbbd5206af951ef70"
job_id = "84438924399202"
debug_job_id = "568069039710705"
cluster_id = "0924-112309-uxg886zj"

# Note: Conversation state is now managed per user session using Gradio's state

def upload_and_process_video(video, analysis_type, debug_mode=False, progress=gr.Progress()):
    """
    Upload video to Unity Catalog volume and process it with the selected analysis type
    Returns only status message, not the video output
    Uses generator function with progress tracking
    """
    try:
        if not video:
            yield None, "❌ Please upload a video first"
            return
        
        if not analysis_type:
            yield None, "❌ Please select an analysis type"
            return
        
        # Hardcoded volume paths
        upload_catalog = "ali_azzouz"
        upload_schema = "football"
        upload_volume = "computer_vision"
        upload_folder = "app"
        
        download_catalog = "ali_azzouz"
        download_schema = "football"
        download_volume = "computer_vision"
        download_folder = "app"
        
        # Get original filename for file extension
        original_filename = os.path.basename(video)
        file_extension = os.path.splitext(original_filename)[1]  # Get .mp4, .avi, etc.
        
        # Phase 1: Upload video (10% of progress)
        progress(0.1, desc="📤 Reading video file...")
        yield None, "📤 Reading video file..."
        
        # Read the uploaded video file
        with open(video, 'rb') as f:
            file_bytes = f.read()
        
        binary_data = io.BytesIO(file_bytes)
        
        # Upload to Unity Catalog volume with fixed filename
        upload_filename = f"uploaded_video{file_extension}"
        upload_volume_file_path = f"/Volumes/{upload_catalog}/{upload_schema}/{upload_volume}/{upload_folder}/{upload_filename}"
        
        progress(0.2, desc="📤 Uploading video to Unity Catalog...")
        yield None, f"📤 Uploading video to: {upload_volume_file_path}"
        
        w.files.upload(upload_volume_file_path, binary_data, overwrite=True)
        
        progress(0.3, desc="✅ Video upload completed")
        yield None, "✅ Video upload completed successfully"
        
        # Phase 2: Start job (20% of progress)
        progress(0.2, desc="🚀 Starting processing job...")
        yield None, "🚀 Starting processing job..."
        
        try:
            # Select job ID based on debug mode
            current_job_id = debug_job_id if debug_mode else job_id
            job_mode = "DEBUG" if debug_mode else "PRODUCTION"
            
            # Start the job with analysis type parameter
            job_parameters = {
                "MODE": analysis_type
            }
            print(f"🚀 Triggering {job_mode} job (ID: {current_job_id}) with parameters: {job_parameters}")
            run_response = w.jobs.run_now(job_id=current_job_id, job_parameters=job_parameters)
            run_id = run_response.run_id
            print(f"✅ {job_mode} job triggered successfully. Run ID: {run_id}")
            
            progress(0.3, desc="✅ Job started successfully")
            yield None, f"✅ {job_mode} job started successfully. Run ID: {run_id}"
            
            # Phase 3: Monitor job progress (30% to 80% of progress)
            import time
            max_wait_time = 1800  # 30 minutes timeout
            check_interval = 30   # Check every 30 seconds
            elapsed_time = 0
            
            print(f"⏳ Starting job monitoring (timeout: {max_wait_time}s, check interval: {check_interval}s)")
            
            while elapsed_time < max_wait_time:
                try:
                    # Calculate progress based on elapsed time (30% to 80%)
                    job_progress = 0.3 + (elapsed_time / max_wait_time) * 0.5
                    progress(job_progress, desc=f"⏳ Monitoring job progress... ({elapsed_time}s/{max_wait_time}s)")
                    
                    run_info = w.jobs.get_run(run_id=run_id)
                    state = run_info.state.life_cycle_state
                    print(f"Job state: {state} {elapsed_time}")
                    
                    # Update status message
                    status_msg = f"⏳ Job Status: {state} (Elapsed: {elapsed_time}s/{max_wait_time}s)"
                    yield None, status_msg
                    
                    # Handle terminal states
                    if state == RunLifeCycleState.TERMINATED:
                        result_state = run_info.state.result_state
                        if result_state == RunResultState.SUCCESS:
                            print(f"🎉 Job completed successfully: {state} after {elapsed_time} seconds")
                            progress(0.8, desc="✅ Job completed successfully")
                            yield None, f"🎉 Job completed successfully after {elapsed_time} seconds"
                            break  # Job completed successfully
                        else:
                            print(f"❌ Job failed with result state: {result_state}")
                            error_message = f"❌ Job failed with result state: {result_state}"
                            if run_info.state.state_message:
                                error_message += f"\nError details: {run_info.state.state_message}"
                            yield None, error_message
                            return
                            
                    elif state == RunLifeCycleState.SKIPPED:
                        print(f"⚠️ Job was skipped")
                        yield None, "❌ Job was skipped"
                        return
                        
                    elif state == RunLifeCycleState.INTERNAL_ERROR:
                        print(f"💥 Job failed with internal error")
                        error_message = f"❌ Job failed with internal error"
                        if run_info.state.state_message:
                            error_message += f"\nError details: {run_info.state.state_message}"
                        yield None, error_message
                        return
                    
                    # Job is still running - wait before next check
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                    
                except Exception as check_error:
                    yield None, f"❌ Error checking job status: {str(check_error)}"
                    return
            
            # If we exit the loop due to timeout
            if elapsed_time >= max_wait_time:
                print(f"⏰ Job timed out after {max_wait_time} seconds")
                yield None, f"❌ Job timed out after {max_wait_time} seconds. Please check the job status manually."
                return
                
        except Exception as job_error:
            print(f"💥 Error starting job: {str(job_error)}")
            yield None, f"❌ Error starting job: {str(job_error)}"
            return

        # Phase 4: Download processed video (80% to 95% of progress)
        processed_filename = f"processed_video{file_extension}"
        download_volume_file_path = f"/Volumes/{download_catalog}/{download_schema}/{download_volume}/{download_folder}/{processed_filename}"
        
        progress(0.8, desc="📥 Downloading processed video...")
        yield None, f"📥 Downloading processed video from: {download_volume_file_path}"
        
        try:
            response = w.files.download(download_volume_file_path)
            processed_video_data = response.contents.read()
            print(f"✅ Video download completed successfully ({len(processed_video_data)} bytes)")
            
            progress(0.9, desc="💾 Saving video locally...")
            yield None, f"✅ Video download completed successfully ({len(processed_video_data)} bytes)"
            
            # OPTION 1: Save to local video folder (current implementation)
            # Create local video folder if it doesn't exist
            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)
            
            # Save to local video folder
            local_video_path = os.path.join(video_folder, processed_filename)
            print(f"💾 Saving processed video to: {local_video_path}")
            with open(local_video_path, 'wb') as f:
                f.write(processed_video_data)
            
            # Get absolute path for Gradio compatibility
            absolute_video_path = os.path.abspath(local_video_path)
            print(f"✅ Processed video saved successfully: {absolute_video_path}")
            
            # Verify file was saved and has content
            if os.path.exists(absolute_video_path) and os.path.getsize(absolute_video_path) > 0:
                file_size = os.path.getsize(absolute_video_path)
                print(f"🎉 Process completed successfully! Final file size: {file_size} bytes")
                
                progress(1.0, desc="🎉 Processing completed successfully!")
                success_message = (
                    f"✅ Processing completed successfully!\n"
                    f"📤 Uploaded: {upload_volume_file_path}\n"
                    f"🔍 Analysis Type: {analysis_type}\n"
                    f"🚀 Job completed successfully!\n"
                    f"🎬 Video has been processed successfully!\n"
                    f"📥 Downloaded: {download_volume_file_path}\n"
                    f"💾 Saved locally: {absolute_video_path}\n"
                    f"📁 File size: {file_size} bytes\n"
                    f"🎯 Click 'Display Output Video' to view the processed video"
                )
                yield None, success_message
            else:
                print(f"❌ Error: Video file verification failed - file doesn't exist or is empty")
                error_message = f"❌ Error: Video file was not saved properly or is empty"
                yield None, error_message
            
        except Exception as download_error:
            # If processed video doesn't exist yet, try to return the original video as fallback
            print(f"❌ Video download failed: {download_error}")
            print(f"🔄 Attempting fallback to original video...")
            
            progress(0.9, desc="🔄 Creating fallback video...")
            yield None, f"❌ Video download failed: {download_error}\n🔄 Attempting fallback to original video..."
            
            # Create local video folder if it doesn't exist
            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)
            
            # OPTION 1: Copy original video to local folder as fallback (current implementation)
            fallback_filename = f"uploaded_video{file_extension}"
            fallback_path = os.path.join(video_folder, fallback_filename)
            
            try:
                # Copy original video to local folder
                print(f"📁 Creating fallback video: {fallback_path}")
                with open(video, 'rb') as src:
                    with open(fallback_path, 'wb') as dst:
                        dst.write(src.read())
                
                absolute_fallback_path = os.path.abspath(fallback_path)
                print(f"✅ Fallback video created successfully: {absolute_fallback_path}")
                
                progress(1.0, desc="✅ Fallback video created")
                upload_success_message = f"✅ Upload successful: {upload_volume_file_path}\n🔍 Analysis Type: {analysis_type}\n🚀 Job completed but processed video not found at: {download_volume_file_path}\n💡 The processed video may not be available yet.\n📁 Fallback video saved: {absolute_fallback_path}\n🎯 Click 'Display Output Video' to view the fallback video"
                yield None, upload_success_message
                
            except Exception as fallback_error:
                print(f"❌ Fallback video creation failed: {fallback_error}")
                upload_success_message = f"✅ Upload successful: {upload_volume_file_path}\n🔍 Analysis Type: {analysis_type}\n🚀 Job completed but processed video not found at: {download_volume_file_path}\n💡 The processed video may not be available yet.\n❌ Could not create fallback video: {fallback_error}"
                yield None, upload_success_message
            
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        yield None, error_message

def display_output_video(analysis_type=None, radar_mode_completed=False):
    """Display the most recent processed video from the videos folder and enable chat only for RADAR mode"""
    try:
        video_folder = "videos"
        if not os.path.exists(video_folder):
            return None, "❌ No videos folder found. Please process a video first.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Process and display a video first", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), False
        
        # Look for processed videos (processed_video.*) first, then fallback videos (uploaded_video.*)
        processed_files = []
        fallback_files = []
        
        for file in os.listdir(video_folder):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                file_path = os.path.join(video_folder, file)
                if file.startswith("processed_video"):
                    processed_files.append((file_path, os.path.getmtime(file_path)))
                elif file.startswith("uploaded_video"):
                    fallback_files.append((file_path, os.path.getmtime(file_path)))
        
        # Prioritize processed videos over fallback videos
        if processed_files:
            latest_file = max(processed_files, key=lambda x: x[1])[0]
            video_type = "processed"
        elif fallback_files:
            latest_file = max(fallback_files, key=lambda x: x[1])[0]
            video_type = "fallback"
        else:
            return None, "❌ No videos found. Please process a video first.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Process and display a video first", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), False
        
        absolute_path = os.path.abspath(latest_file)
        
        if os.path.exists(absolute_path) and os.path.getsize(absolute_path) > 0:
            file_size = os.path.getsize(absolute_path)
            
            # Check if analysis type is RADAR to enable chat
            if analysis_type == "RADAR":
                success_message = f"✅ Displaying latest {video_type} video:\n📁 File: {os.path.basename(absolute_path)}\n💾 Size: {file_size} bytes\n📂 Path: {absolute_path}\n🔍 Analysis: {analysis_type}\n\n💬 Chat is now available! Ask questions about the video analytics."
                return absolute_path, success_message, gr.update(interactive=True, placeholder="Type a message and press Enter...", value="", label="Message", scale=4), gr.update(interactive=True), gr.update(interactive=True), "🟢 Chat Enabled - Ready to answer questions about the video analytics", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(placeholder="Chat is now available! Ask questions about the video analytics.", value=None), True  # Set radar_mode_completed to True
            else:
                success_message = f"✅ Displaying latest {video_type} video:\n📁 File: {os.path.basename(absolute_path)}\n💾 Size: {file_size} bytes\n📂 Path: {absolute_path}\n🔍 Analysis: {analysis_type}\n\n💬 Chat is only available for RADAR analysis mode."
                return absolute_path, success_message, gr.update(interactive=False, placeholder="Chat only available for RADAR analysis..."), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Only available for RADAR analysis", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), radar_mode_completed  # Keep current state
        else:
            return None, "❌ Video file is empty or corrupted.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Process and display a video first", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), False
            
    except Exception as e:
        return None, f"❌ Error displaying video: {str(e)}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Process and display a video first", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), False

def process_video_and_enable_display(video, analysis_type, debug_mode=False, progress=gr.Progress()):
    """Process video and enable display button if successful"""
    # Use the generator function to get the final result
    final_status = None
    for _, status_message in upload_and_process_video(video, analysis_type, debug_mode, progress):
        final_status = status_message
        # Yield intermediate status updates
        yield status_message, gr.update(interactive=False), gr.update(interactive=False, placeholder="Chat disabled - Process and display a video in RADAR mode first..."), analysis_type  # Keep both display button and chat disabled, store analysis type
    
    # Check if processing was successful by looking for success indicators in the message
    if final_status and ("✅ Processing completed successfully!" in final_status or "✅ Upload successful:" in final_status):
        yield final_status, gr.update(interactive=True), gr.update(interactive=False, placeholder="Chat disabled - Process and display a video in RADAR mode first..."), analysis_type  # Enable display button, keep chat disabled, store analysis type
    else:
        yield final_status, gr.update(interactive=False), gr.update(interactive=False, placeholder="Chat disabled - Process and display a video in RADAR mode first..."), analysis_type  # Keep both disabled, store analysis type


def disable_display_button_on_video_upload(video, current_analysis_type=None, radar_mode_completed=False):
    """Disable display button and chat when a new video is uploaded"""
    if video:
        # If RADAR mode has been completed at least once, keep chat enabled
        if radar_mode_completed:
            return gr.update(interactive=False), gr.update(interactive=True, placeholder="Type a message and press Enter..."), current_analysis_type
        else:
            return gr.update(interactive=False), gr.update(interactive=False, placeholder="Chat disabled - Process and display a video in RADAR mode first..."), None  # Disable both display button and chat, reset analysis type
    else:
        return gr.update(interactive=False), gr.update(interactive=False, placeholder="Chat disabled - Process and display a video in RADAR mode first..."), None  # Keep both disabled, reset analysis type

def get_cluster_status_message():
    """Get cluster status message for UI display"""
    try:
        cluster_info = w.clusters.get(cluster_id=cluster_id)
        state = cluster_info.state
        return f"🟢 Cluster Status: {state}"
    except Exception as e:
        return f"❌ Cluster Status: Error - {str(e)}"

def test_genie_connection():
    """Test Genie connection and return status"""
    try:
        print(f"🧪 Testing Genie connection with space ID: {genie_space_id}")
        # Try to start a simple conversation
        conversation = w.genie.start_conversation_and_wait(genie_space_id, "Hello")
        print(f"✅ Genie connection successful. Conversation ID: {conversation.conversation_id}")
        return f"🟢 Genie Status: Connected (Space ID: {genie_space_id})"
    except Exception as e:
        print(f"❌ Genie connection failed: {str(e)}")
        return f"❌ Genie Status: Error - {str(e)}"

def get_query_result(statement_id):
    """Get query result as DataFrame"""
    try:
        result = w.statement_execution.get_statement(statement_id)
        return pd.DataFrame(
            result.result.data_array, 
            columns=[i.name for i in result.manifest.schema.columns]
        )
    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to fetch query result: {str(e)}"]})

def process_genie_response(response):
    """Process Genie response and format for Gradio display"""
    messages = []
    
    for i in response.attachments:
        if i.text:
            message = {"role": "assistant", "content": i.text.content}
            messages.append(message)
        elif i.query:
            try:
                data = get_query_result(response.query_result.statement_id)
                message = {
                    "role": "assistant", 
                    "content": f"{i.query.description}\n\nQuery executed successfully. Results:\n{data.to_string()}", 
                    "code": i.query.query
                }
                messages.append(message)
            except Exception as e:
                message = {
                    "role": "assistant", 
                    "content": f"{i.query.description}\n\nQuery: {i.query.query}\n\nError executing query: {str(e)}"
                }
                messages.append(message)
    
    return messages

def chat_response(history, message, conversation_id_state):
    """AI-powered chatbot using Databricks Genie"""
    if history is None:
        history = []
    
    # Initialize conversation_id_state if it doesn't exist
    if conversation_id_state is None:
        conversation_id_state = None
    
    try:
        # Check if we have an existing conversation
        conversation_id = conversation_id_state
        
        print(f"🤖 Genie Chat - Message: {message}")
        print(f"🤖 Genie Chat - Conversation ID: {conversation_id}")
        
        if conversation_id:
            # Continue existing conversation
            print(f"🔄 Continuing existing conversation: {conversation_id}")
            conversation = w.genie.create_message_and_wait(
                genie_space_id, conversation_id, message
            )
        else:
            # Start new conversation
            print(f"🆕 Starting new conversation")
            conversation = w.genie.start_conversation_and_wait(genie_space_id, message)
            conversation_id_state = conversation.conversation_id
            print(f"✅ New conversation created: {conversation.conversation_id}")
        
        print(f"📝 Conversation response received")
        
        # Process the response
        genie_messages = process_genie_response(conversation)
        print(f"📋 Processed {len(genie_messages)} messages from Genie")
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        # Add AI responses to history (only if there are actual responses)
        if genie_messages:
            for genie_msg in genie_messages:
                if genie_msg["role"] == "assistant":
                    content = genie_msg["content"]
                    if "code" in genie_msg:
                        content += f"\n\n```sql\n{genie_msg['code']}\n```"
                    # Add assistant response
                    history.append({"role": "assistant", "content": content})
        else:
            # If no messages from Genie, show a default response
            history.append({"role": "assistant", "content": "I'm processing your request..."})
        
        return history, "", get_conversation_status(conversation_id_state), conversation_id_state
        
    except Exception as e:
        print(f"❌ Genie Chat Error: {str(e)}")
        
        # Reset conversation ID if there's an error
        conversation_id_state = None
        
        # Provide a helpful fallback response
        if "MessageStatus.FAILED" in str(e):
            error_response = (
                "❌ Genie Assistant is currently unavailable. This could be due to:\n"
                "• Genie service being temporarily down\n"
                "• Invalid Genie space ID or permissions\n"
                "• Network connectivity issues\n\n"
                "Please try again later or contact your administrator."
            )
        else:
            error_response = f"❌ Error: {str(e)}"
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_response})
        return history, "", get_conversation_status(conversation_id_state), conversation_id_state


def reset_conversation(conversation_id_state):
    """Reset the conversation ID and clear chat history to start a new session"""
    conversation_id_state = None
    print("🔄 Conversation reset - starting fresh session")
    
    # Create a reset message in chat history
    reset_message = "🔄 Conversation reset! Starting a new session."
    history = [{"role": "assistant", "content": reset_message}]
    
    return history, "", get_conversation_status(conversation_id_state), conversation_id_state


def get_conversation_status(conversation_id):
    """Get current conversation status for display"""
    if conversation_id:
        return f"🟢 Active Session: {conversation_id[:8]}..."
    else:
        return "⚪ No Active Session"

def update_job_mode_status(debug_mode):
    """Update job mode status display based on debug toggle"""
    if debug_mode:
        return f"🐛 Debug Mode - Job ID: {debug_job_id}"
    else:
        return f"🟢 Production Mode - Job ID: {job_id}"

def download_sample_video(sample_filename):
    """Download a sample video from Unity Catalog volume and return local path"""
    try:
        if not sample_filename:
            return None, "❌ Please select a sample video"
        
        # Unity Catalog volume paths for samples
        samples_catalog = "ali_azzouz"
        samples_schema = "football"
        samples_volume = "computer_vision"
        samples_folder = "samples"
        
        # Download from Unity Catalog volume
        download_volume_file_path = f"/Volumes/{samples_catalog}/{samples_schema}/{samples_volume}/{samples_folder}/{sample_filename}"
        
        print(f"📥 Downloading sample video from: {download_volume_file_path}")
        
        try:
            response = w.files.download(download_volume_file_path)
            sample_video_data = response.contents.read()
            print(f"✅ Sample video download completed successfully ({len(sample_video_data)} bytes)")
            
            # Create local video folder if it doesn't exist
            video_folder = "videos"
            os.makedirs(video_folder, exist_ok=True)
            
            # Save to local video folder
            local_video_path = os.path.join(video_folder, sample_filename)
            print(f"💾 Saving sample video to: {local_video_path}")
            with open(local_video_path, 'wb') as f:
                f.write(sample_video_data)
            
            # Get absolute path for Gradio compatibility
            absolute_video_path = os.path.abspath(local_video_path)
            print(f"✅ Sample video saved successfully: {absolute_video_path}")
            
            # Verify file was saved and has content
            if os.path.exists(absolute_video_path) and os.path.getsize(absolute_video_path) > 0:
                file_size = os.path.getsize(absolute_video_path)
                success_message = f"✅ Sample video loaded successfully!\n📁 File: {sample_filename}\n💾 Size: {file_size} bytes\n📂 Path: {absolute_video_path}"
                return absolute_video_path, success_message
            else:
                error_message = f"❌ Error: Sample video file was not saved properly or is empty"
                return None, error_message
            
        except Exception as download_error:
            print(f"❌ Sample video download failed: {download_error}")
            error_message = f"❌ Error downloading sample video: {download_error}"
            return None, error_message
            
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        return None, error_message

def demo_radar_analysis():
    """Demo function that downloads uploaded_video and processed_video from samples folder and enables chat"""
    try:
        # Unity Catalog volume paths for samples
        samples_catalog = "ali_azzouz"
        samples_schema = "football"
        samples_volume = "computer_vision"
        samples_folder = "samples"
        
        # Create local video folder if it doesn't exist
        video_folder = "videos"
        os.makedirs(video_folder, exist_ok=True)
        
        # Download uploaded_video.mp4
        uploaded_video_path = f"/Volumes/{samples_catalog}/{samples_schema}/{samples_volume}/{samples_folder}/uploaded_video.mp4"
        print(f"📥 Downloading uploaded video from: {uploaded_video_path}")
        
        try:
            response = w.files.download(uploaded_video_path)
            uploaded_video_data = response.contents.read()
            print(f"✅ Uploaded video download completed successfully ({len(uploaded_video_data)} bytes)")
            
            # Save uploaded video locally
            local_uploaded_path = os.path.join(video_folder, "uploaded_video.mp4")
            with open(local_uploaded_path, 'wb') as f:
                f.write(uploaded_video_data)
            
            absolute_uploaded_path = os.path.abspath(local_uploaded_path)
            print(f"✅ Uploaded video saved successfully: {absolute_uploaded_path}")
            
        except Exception as upload_error:
            print(f"❌ Uploaded video download failed: {upload_error}")
            return None, None, f"❌ Error downloading uploaded video: {upload_error}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Demo failed", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), None, gr.update(interactive=False), None
        
        # Download processed_video.mp4
        processed_video_path = f"/Volumes/{samples_catalog}/{samples_schema}/{samples_volume}/{samples_folder}/processed_video.mp4"
        print(f"📥 Downloading processed video from: {processed_video_path}")
        
        try:
            response = w.files.download(processed_video_path)
            processed_video_data = response.contents.read()
            print(f"✅ Processed video download completed successfully ({len(processed_video_data)} bytes)")
            
            # Save processed video locally
            local_processed_path = os.path.join(video_folder, "processed_video.mp4")
            with open(local_processed_path, 'wb') as f:
                f.write(processed_video_data)
            
            absolute_processed_path = os.path.abspath(local_processed_path)
            print(f"✅ Processed video saved successfully: {absolute_processed_path}")
            
        except Exception as processed_error:
            print(f"❌ Processed video download failed: {processed_error}")
            return absolute_uploaded_path, None, f"✅ Uploaded video loaded successfully!\n❌ Error downloading processed video: {processed_error}", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Demo partially failed", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), None, gr.update(interactive=False), None
        
        # Both videos downloaded successfully
        uploaded_size = os.path.getsize(absolute_uploaded_path)
        processed_size = os.path.getsize(absolute_processed_path)
        
        success_message = (
            f"🎉 Demo RADAR Analysis loaded successfully!\n"
            f"📤 Input Video: uploaded_video.mp4 ({uploaded_size} bytes)\n"
            f"📥 Output Video: processed_video.mp4 ({processed_size} bytes)\n"
            f"🔍 Analysis Type: RADAR\n"
            f"💬 Chat is now enabled! Ask questions about the video analytics.\n"
            f"📂 Input Path: {absolute_uploaded_path}\n"
            f"📂 Output Path: {absolute_processed_path}"
        )
        
        print(f"🎉 Demo setup completed successfully!")
        print(f"🔧 Debug: About to return gr.update for msg component")
        
        # Create explicit gr.update objects
        msg_update = gr.update(interactive=True, placeholder="Type a message and press Enter...", value="", label="Message", scale=4)
        send_update = gr.update(interactive=True)
        reset_update = gr.update(interactive=True)
        chatbot_update = gr.update(placeholder="Chat is now available! Ask questions about the video analytics.", value=None)
        display_update = gr.update(interactive=True)
        
        print(f"🔧 Debug: Created gr.update objects")
        
        # Return the videos and directly enable chat since this is RADAR mode
        return (
            absolute_uploaded_path,  # video_input
            absolute_processed_path,  # video_output
            success_message,  # status_output
            msg_update,  # msg
            send_update,  # send_button
            reset_update,  # reset_button
            "🟢 Chat Enabled - Demo RADAR analysis ready!",  # conversation_status
            gr.update(interactive=True),  # example_btn_1
            gr.update(interactive=True),  # example_btn_2
            gr.update(interactive=True),  # example_btn_3
            gr.update(interactive=True),  # example_btn_4
            chatbot_update,  # chatbot
            "RADAR",  # current_analysis_type
            display_update,  # display_button
            "RADAR",  # analysis_type_dropdown
            True  # radar_mode_completed - Set to True since demo completed RADAR mode
        )
        
    except Exception as e:
        error_message = f"❌ Demo Error: {str(e)}"
        print(f"❌ Demo failed: {error_message}")
        return None, None, error_message, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), "⚪ Chat Disabled - Demo failed", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(placeholder="Chat will be available after you process and display a video in RADAR mode"), None, gr.update(interactive=False), None, False

def send_example_question(question, history, conversation_id_state):
    """Send an example question to the chat"""
    if history is None:
        history = []
    
    # Add the question to the message input and trigger chat response
    return question, history, conversation_id_state

def enable_example_buttons():
    """Enable example question buttons when chat is available"""
    return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def disable_example_buttons():
    """Disable example question buttons when chat is not available"""
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

# Create Gradio interface
with gr.Blocks(title="Football Computer Vision Analytics", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚽🎥 Football Computer Vision Analytics using Roboflow Supervision")
    gr.Markdown("Upload football videos, run a computer vision analysis and ask your questions in the game below!")
    
    # Progress tracking info
    with gr.Row():
        gr.Markdown("""
        ### 📊 [Access the dashboard](https://e2-demo-field-eng.cloud.databricks.com/dashboardsv3/01f09ae460f1144a93b78db88151c7c8/published?o=1444828305810485)
        ### 🤖 [Access the model in Unity Catalog](https://e2-demo-field-eng.cloud.databricks.com/explore/data/models/ali_azzouz/football/football_cv?o=1444828305810485)
        ### 🤖 [Access Roboflow Supervision example](https://github.com/roboflow/sports/tree/main/examples/soccer)
        ### TO DO: 
            - Deploy model on serving endpoint and yield image per image instead of uploading/downloading
            - Explore streaming inference
            - Optimize job inference using AIR GPU
            - Optimize uploading/downloading videos from temporary memory
            - FFMEPG direct codec
            - Improve detection for data collection
            - Use unique ID for each video and append data instead of overwrite
        """)
    
    # Status indicators
    with gr.Row():
        cluster_status = gr.Textbox(
            label="🖥️ Cluster Status",
            value=get_cluster_status_message(),
            interactive=False,
            show_copy_button=True
        )
        genie_status = gr.Textbox(
            label="🤖 Genie Status",
            value=test_genie_connection(),
            interactive=False,
            show_copy_button=True
        )
    
    with gr.Row():
        refresh_cluster_btn = gr.Button("🔄 Refresh Cluster", size="sm")
        test_genie_btn = gr.Button("🧪 Test Genie", size="sm")
        debug_toggle = gr.Checkbox(label="🐛 Debug Mode", value=False)
    
    # Job mode status
    with gr.Row():
        job_mode_status = gr.Textbox(
            label="🔧 Job Mode Status",
            value="🟢 Production Mode - Job ID: 84438924399202",
            interactive=False,
            show_copy_button=True
        )
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📹 Upload Video", height=300)
            
            # Sample video dropdown
            sample_video_dropdown = gr.Dropdown(
                choices=[
                    "08fd33_0.mp4",
                    "0bfacc_0.mp4", 
                    "121364_0.mp4",
                    "2e57b9_0.mp4",
                    "573e61_0.mp4"
                ],
                label="📁 Load Sample Video",
                value=None,
                info="Select a sample video to load from Unity Catalog",
                allow_custom_value=False
            )
            
            analysis_type_dropdown = gr.Dropdown(
                choices=[
                    "PITCH_DETECTION",
                    "PLAYER_DETECTION", 
                    "BALL_DETECTION",
                    "PLAYER_TRACKING",
                    "TEAM_CLASSIFICATION",
                    "RADAR"
                ],
                label="🔍 Select Analysis Type",
                value="BALL_DETECTION",
                info="Choose the type of computer vision analysis to perform"
            )
            
            process_button = gr.Button("🚀 Upload & Process Video", variant="primary", size="lg")
            
            # Progress bar for video processing
            progress_bar = gr.Progress()
            
            display_button = gr.Button("📺 Display Output Video", variant="secondary", size="lg", interactive=False)
            
        with gr.Column():
            video_output = gr.Video(label="📺 Processed Video Output", height=300)
            
            # Demo button
            demo_button = gr.Button("🎬 Demo RADAR Analysis", variant="secondary", size="lg")
            
            status_output = gr.Textbox(
                label="📋 Status",
                lines=6,
                max_lines=15,
                #interactive=False,
                show_copy_button=True,
                placeholder="Status updates will appear here during video processing..."
            )
    
    # Connect the refresh cluster status button
    refresh_cluster_btn.click(
        get_cluster_status_message,
        inputs=[],
        outputs=[cluster_status]
    )
    
    # Connect the test genie button
    test_genie_btn.click(
        test_genie_connection,
        inputs=[],
        outputs=[genie_status]
    )
    
    # Connect the debug toggle
    debug_toggle.change(
        update_job_mode_status,
        inputs=[debug_toggle],
        outputs=[job_mode_status]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 💬 Genie Assistant")
    gr.Markdown("💡 Chat is only available after processing and displaying a video in RADAR mode")
    
    # Conversation status
    with gr.Row():
        conversation_status = gr.Textbox(
            label="💬 Chat Session Status",
            value="⚪ Chat Disabled - Process and display a video in RADAR mode first",
            interactive=False,
            show_copy_button=True
        )
    
    # Session state for conversation ID and analysis type
    conversation_id_state = gr.State(value=None)
    current_analysis_type = gr.State(value=None)
    radar_mode_completed = gr.State(value=False)  # Track if RADAR mode has been completed at least once
    
    with gr.Row():
        chatbot = gr.Chatbot(
            height=300, 
            show_copy_button=True, 
            type="messages",
            placeholder="Chat will be available after you process and display a video in RADAR mode"
        )
        
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Chat disabled - Process and display a video in RADAR mode first...",
            label="Message",
            scale=4,
            interactive=False
        )
        send_button = gr.Button("Send", scale=1, interactive=False)
        reset_button = gr.Button("🔄 Reset Chat", scale=1, variant="secondary", interactive=False)
    
    # Example questions section
    gr.Markdown("#### 📝 Example Questions You Can Ask:")
    
    # Example question buttons
    with gr.Row():
        example_btn_1 = gr.Button("🎯 Max shots by player", size="sm", interactive=False)
        example_btn_2 = gr.Button("⚡ Average player speed", size="sm", interactive=False)
        example_btn_3 = gr.Button("📊 Event types", size="sm", interactive=False)
        example_btn_4 = gr.Button("📈 Team possession %", size="sm", interactive=False)
    
    # Handle chat interactions
    msg.submit(chat_response, [chatbot, msg, conversation_id_state], [chatbot, msg, conversation_status, conversation_id_state])
    send_button.click(chat_response, [chatbot, msg, conversation_id_state], [chatbot, msg, conversation_status, conversation_id_state])
    
    # Handle reset conversation
    reset_button.click(
        reset_conversation,
        inputs=[conversation_id_state],
        outputs=[chatbot, msg, conversation_status, conversation_id_state]
    )
    
    # Connect the process button
    process_button.click(
        process_video_and_enable_display,
        inputs=[video_input, analysis_type_dropdown, debug_toggle],
        outputs=[status_output, display_button, msg, current_analysis_type]  # Include chat input and analysis type in outputs
    )
    
    # Disable display button when video is uploaded
    video_input.change(
        disable_display_button_on_video_upload,
        inputs=[video_input, current_analysis_type, radar_mode_completed],
        outputs=[display_button, msg, current_analysis_type]  # Include chat input and analysis type in outputs
    )
    
    # Connect the display button
    display_button.click(
        display_output_video,
        inputs=[current_analysis_type, radar_mode_completed],
        outputs=[video_output, status_output, msg, send_button, reset_button, conversation_status, example_btn_1, example_btn_2, example_btn_3, example_btn_4, chatbot, radar_mode_completed]  # Enable chat components and example buttons
    )
    
    # Connect the sample video dropdown
    sample_video_dropdown.change(
        download_sample_video,
        inputs=[sample_video_dropdown],
        outputs=[video_input, status_output]
    )
    
    # Connect the demo button
    demo_button.click(
        demo_radar_analysis,
        inputs=[],
        outputs=[video_input, video_output, status_output, msg, send_button, reset_button, conversation_status, example_btn_1, example_btn_2, example_btn_3, example_btn_4, chatbot, current_analysis_type, display_button, analysis_type_dropdown, radar_mode_completed]
    )
    
    # Connect example question buttons
    example_btn_1.click(
        lambda history, conversation_id_state: chat_response(history, "What is the maximum number of shots taken by a player in matches?", conversation_id_state),
        inputs=[chatbot, conversation_id_state],
        outputs=[chatbot, msg, conversation_status, conversation_id_state]
    )
    example_btn_2.click(
        lambda history, conversation_id_state: chat_response(history, "What is the average speed of players during the matches?", conversation_id_state),
        inputs=[chatbot, conversation_id_state],
        outputs=[chatbot, msg, conversation_status, conversation_id_state]
    )
    example_btn_3.click(
        lambda history, conversation_id_state: chat_response(history, "What are the different event types that players participated in?", conversation_id_state),
        inputs=[chatbot, conversation_id_state],
        outputs=[chatbot, msg, conversation_status, conversation_id_state]
    )
    example_btn_4.click(
        lambda history, conversation_id_state: chat_response(history, "What is the weekly average possession percentage of each team?", conversation_id_state),
        inputs=[chatbot, conversation_id_state],
        outputs=[chatbot, msg, conversation_status, conversation_id_state]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
