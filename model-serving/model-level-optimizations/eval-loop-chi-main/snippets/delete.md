
::: {.cell .markdown}

## Delete resources

When we are finished, we must delete the VM server instance to make the resources available to other users.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected. 

:::

::: {.cell .code}
```python
from chi import server, context
import chi, os, time, datetime

context.version = "1.0" 
context.choose_project()
context.choose_site(default="KVM@TACC")
```
:::


::: {.cell .code}
```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.get_server(f"node-eval-loop-{username}")
s.delete()
```
:::

