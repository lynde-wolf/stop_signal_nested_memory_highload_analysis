from stop_wm.serverExperimentalData import ServerExperimentalData


def test_ServerExperimentalData():
    sed = ServerExperimentalData()
    assert sed.subject is None
