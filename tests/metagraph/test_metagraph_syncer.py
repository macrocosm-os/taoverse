import threading
import unittest
from unittest import mock

import bittensor as bt

from taoverse.metagraph.metagraph_syncer import MetagraphSyncer


class TestMetagraphSyncer(unittest.TestCase):
    def test_do_initial_sync(self):
        # Mock subtensor.metagraph() function
        metagraph1 = bt.metagraph(netuid=1, sync=False)
        metagraph2 = bt.metagraph(netuid=2, sync=False)

        def get_metagraph(netuid, lite) -> bt.metagraph:
            if netuid == 1:
                return metagraph1
            elif netuid == 2:
                return metagraph2
            else:
                raise Exception("Invalid netuid")

        mock_subtensor = mock.MagicMock(spec=bt.subtensor)
        metagraph_mock = mock.MagicMock(side_effect=get_metagraph)
        mock_subtensor.metagraph = metagraph_mock

        # Create MetagraphSyncer instance with mock subtensor
        metagraph_syncer = MetagraphSyncer(mock_subtensor, {1: 1, 2: 1})

        # Call do_initial_sync method
        metagraph_syncer.do_initial_sync()

        # Verify get_metagraph() returns the expected metagraph.
        # We can't check object equality because of how equality is done on bt.metagraph
        # so just check the netuid.
        self.assertEqual(metagraph_syncer.get_metagraph(1).netuid, metagraph1.netuid)
        self.assertEqual(metagraph_syncer.get_metagraph(2).netuid, metagraph2.netuid)

    def test_listener_called(self):
        # Mock subtensor.metagraph() function
        metagraph1 = bt.metagraph(netuid=1, sync=False)
        metagraph2 = bt.metagraph(netuid=2, sync=False)

        def get_metagraph(netuid, lite) -> bt.metagraph:
            if netuid == 1:
                return metagraph1
            elif netuid == 2:
                return metagraph2
            else:
                raise Exception("Invalid netuid")

        mock_subtensor = mock.MagicMock(spec=bt.subtensor)
        metagraph_mock = mock.MagicMock(side_effect=get_metagraph)
        mock_subtensor.metagraph = metagraph_mock

        # Create MetagraphSyncer instance with mock subtensor
        metagraph_syncer = MetagraphSyncer(mock_subtensor, {1: 1, 2: 1})

        # Call do_initial_sync method
        metagraph_syncer.do_initial_sync()

        # Register a listener for netuid 1.
        event = threading.Event()

        def listener(metagraph, netuid):
            self.assertEqual(metagraph.netuid, 1)
            self.assertEqual(netuid, 1)
            event.set()

        metagraph_syncer.register_listener(listener, [1])

        # Since we sync every 1 second, verify the listener is called within 5 seconds.
        event.wait(5)

    def test_lite_metagraph(self):
        # Mock subtensor.metagraph() function
        metagraph1 = bt.metagraph(netuid=1, sync=False)

        def get_metagraph(netuid, lite) -> bt.metagraph:
            if netuid == 1 and lite == True:
                return metagraph1
            else:
                raise Exception("Invalid netuid")

        mock_subtensor = mock.MagicMock(spec=bt.subtensor)
        metagraph_mock = mock.MagicMock(side_effect=get_metagraph)
        mock_subtensor.metagraph = metagraph_mock

        # Create MetagraphSyncer instance with mock subtensor
        metagraph_syncer = MetagraphSyncer(
            mock_subtensor,
            {
                1: 1,
            },
        )

        # Call do_initial_sync method
        metagraph_syncer.do_initial_sync()

        # Verify get_metagraph() returns the expected metagraph.
        self.assertEqual(metagraph_syncer.get_metagraph(1).netuid, metagraph1.netuid)

    def test_non_lite_metagraph(self):
        # Mock subtensor.metagraph() function
        metagraph1 = bt.metagraph(netuid=1, sync=False)

        def get_metagraph(netuid, lite) -> bt.metagraph:
            if netuid == 1 and lite == False:
                return metagraph1
            else:
                raise Exception("Invalid netuid")

        mock_subtensor = mock.MagicMock(spec=bt.subtensor)
        metagraph_mock = mock.MagicMock(side_effect=get_metagraph)
        mock_subtensor.metagraph = metagraph_mock

        # Create MetagraphSyncer instance with mock subtensor
        metagraph_syncer = MetagraphSyncer(mock_subtensor, {1: 1}, lite=False)

        # Call do_initial_sync method
        metagraph_syncer.do_initial_sync()

        # Verify get_metagraph() returns the expected metagraph.
        self.assertEqual(metagraph_syncer.get_metagraph(1).netuid, metagraph1.netuid)


if __name__ == "__main__":
    unittest.main()
