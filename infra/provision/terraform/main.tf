provider "openstack" {
  auth_url    = "https://chi.tacc.chameleoncloud.org:5000/v3"
  tenant_name = "<your-project-id>"
  domain_name = "default"
  user_name   = "<your-username>"
  password    = "<your-password>"
  region      = "CHI@TACC"
}

resource "openstack_compute_keypair_v2" "keypair" {
  name       = "mlops-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "openstack_compute_instance_v2" "mlops_vm" {
  name            = "mlops-vm"
  image_name      = "CC-Ubuntu22.04"
  flavor_name     = "m1.medium"
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = ["default"]

  network {
    name = "sharednet1"
  }
}
