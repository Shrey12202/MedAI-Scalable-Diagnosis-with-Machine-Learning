# Persistent storage on Chameleon

In this tutorial, we will practice using two types of persistent storage options on Chameleon:

* object storage, which you may use to e.g. store large training data sets
* and block storage, which you may use for persistent storage for services that run on VM instances (e.g. MLFlow, Prometheus, etc.)

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You should also have added your SSH key to the KVM@TACC site and the CHI@TACC site.

Follow along at [Persistent storage on Chameleon](https://teaching-on-testbeds.github.io/data-persist-chi/).

This tutorial uses: one `m1.large` VM at KVM@TACC, and one floating IP, one 2 GiB block storage volume at KVM@TACC, and one object store container at CHI@TACC.

---

This material is based upon work supported by the National Science Foundation under Grant No. 2230079.

