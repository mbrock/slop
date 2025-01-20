from tagflow import tag, text


def upload_area():
    """Create an inviting upload area for interview files"""
    with tag.div(
        classes="max-w-md mx-auto",
        id="upload-zone",
    ):
        with tag.form(
            action="/upload",
            method="post",
            enctype="multipart/form-data",
        ):
            with tag.div(
                classes=(
                    "w-16 h-16 mx-auto border-2 border-stone-400 bg-stone-200 rounded-lg "
                    "flex items-center justify-center "
                    "transition-colors duration-200 ease-in-out "
                    "hover:border-blue-400 hover:bg-blue-50 "
                    "cursor-pointer relative"
                ),
            ):
                with tag.div(classes="text-gray-400 text-4xl font-light"):
                    text("+")

                with tag.input(
                    type="file",
                    name="audio",
                    accept="audio/*",
                    classes=(
                        "absolute inset-0 w-full h-full opacity-0 cursor-pointer "
                        "file:cursor-pointer"
                    ),
                    onchange="this.form.submit()",
                ):
                    pass
