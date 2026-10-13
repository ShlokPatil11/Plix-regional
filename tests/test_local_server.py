def test_local_server_exposes_the_application():
    import local_server

    assert local_server.app.title == "Directory Language Studio"
