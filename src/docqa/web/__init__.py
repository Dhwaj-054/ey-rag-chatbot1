"""Stage 5, the chat screen: a small local web server and a page that talks to it.

    python -m docqa.web

``server.py`` holds the server (Starlette, run by uvicorn). ``static/`` holds the page:
plain HTML, CSS and JavaScript, with nothing fetched from outside this machine.
"""
