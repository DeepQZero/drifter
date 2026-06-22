run `initial_trainer.py`; model id can be changed at bottom of script

individual models can be tested in `test.py`; make sure to change the 
curriculum number.

combined models can be tested in `full_test.py`; place your models at the top




### Troubleshooting
- Action isn't set to deterministic for testing
- Agent is docking and not drifting to dock (rewards are smaller for 
  docking than drifting)


### Ideas 
- Reduce entropy coefficient
- Make POMDP and remove time step
- Curricula could be more focused at start states (e.g. near 2.5, 10, 50, 150)

### TODO
- Do env.unwrapped