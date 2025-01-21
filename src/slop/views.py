from tagflow import tag, text


def upload_area(target: str):
    """Create an inviting upload area for interview files with HTMX upload progress"""
    with tag.div(
        classes="max-w-md mx-auto",
        id="upload-zone",
    ):
        with tag.form(
            id="upload-form",
            classes="relative",
            **{
                "hx-encoding": "multipart/form-data",
                "hx-post": "/upload",
                "hx-target": target,
                "hx-trigger": "change from:(find input)",
            },
        ):
            # Upload area
            with tag.div(
                classes=(
                    "w-full h-32 border-2 border-gray-400 bg-gray-200 rounded-lg "
                    "flex flex-col items-center justify-center gap-2 "
                    "transition-colors duration-200 ease-in-out "
                    "hover:border-blue-400 hover:bg-blue-50 "
                    "cursor-pointer relative"
                ),
            ):
                # Upload icon and text
                with tag.div(classes="text-gray-400 text-center"):
                    with tag.div(classes="text-4xl font-light"):
                        text("+")
                    with tag.div(classes="text-sm mt-1"):
                        text("Click to upload audio file")

                # File input
                with tag.input(
                    type="file",
                    name="audio",
                    accept="audio/*",
                    classes=(
                        "absolute inset-0 w-full h-full opacity-0 cursor-pointer "
                        "file:cursor-pointer"
                    ),
                ):
                    pass

            # Progress bar (hidden by default)
            with tag.div(
                id="progress-container",
                classes="mt-4 hidden",
            ):
                with tag.div(classes="flex justify-between text-sm text-gray-600 mb-1"):
                    with tag.span():
                        text("Uploading...")
                    with tag.span(id="progress-text"):
                        text("0%")
                with tag.div(classes="w-full bg-gray-200 rounded-full h-2"):
                    with tag.div(
                        id="progress-bar",
                        classes="bg-blue-600 rounded-full h-2 transition-all",
                        style="width: 0%",
                    ):
                        pass

        # Progress handling script
        with tag.script():
            text("""
                htmx.on('#upload-form', 'htmx:xhr:progress', function(evt) {
                    // Show progress container
                    htmx.find('#progress-container').classList.remove('hidden');
                    
                    // Calculate percentage
                    var percent = Math.round((evt.detail.loaded / evt.detail.total) * 100);
                    
                    // Update progress bar and text
                    htmx.find('#progress-bar').style.width = percent + '%';
                    htmx.find('#progress-text').innerText = percent + '%';
                });
                
                htmx.on('#upload-form', 'htmx:beforeRequest', function(evt) {
                    // Show progress container when upload starts
                    htmx.find('#progress-container').classList.remove('hidden');
                });
            """)
