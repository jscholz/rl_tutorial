repo7 -- Python code submission from Jonathan Scholz
==============

This project contains a python implementation of the cliff-world problem as described by Sutton & Barto:
http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html

**This project is structured as follows:**
- *agents.py*: Contains definitions of the core learning algorithms (SARSA and Q-Learning)
- *domains.py*: Contains an implementation of the Cliff-World domain and a Tkinter-based visualization
- *experiments.py*: Contains the helper functions for executing RL experiments.  This is the main point-of-entry
- *plot_rewards.py*: Contains plotting functions for visualizing results
- *unit_tests.py*: Contains a set of unit tests for the main classes and data structures

--------------

**Dependencies:**
	python 2.7
	numpy (developed with v1.8.0.dev-9597b1f)
	scipy (developed with v0.11.0)
	Tkinter (developed with v81008)

This project was developed on a 2013 macbook pro running Mac OS 10.9.4

--------------

**Instructions:** 

- To generate the main graph comparing SARSA and Q-Learning:
	$ python experiments.py [--render]

To generate interact with the cliff-domain using the keyboard interface
	$ python experiments.py --keyboard --render

- To run unit tests:
	$ python unit_tests.py

- To visualize the cliff domain in tkinter and ascii:
	$ python domains.py