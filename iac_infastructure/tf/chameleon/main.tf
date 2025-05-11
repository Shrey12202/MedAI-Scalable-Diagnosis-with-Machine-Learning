# Define Security Groups (if they don't already exist)
resource "openstack_networking_secgroup_v2" "allow_ssh" {
  name        = "allow-ssh"
  description = "Allow SSH access"
}

resource "openstack_networking_secgroup_v2" "allow_9001" {
  name        = "allow-9001"
  description = "Allow access to port 9001"
}

resource "openstack_networking_secgroup_v2" "allow_8000" {
  name        = "allow-8000"
  description = "Allow access to port 8000"
}

resource "openstack_networking_secgroup_v2" "allow_8080" {
  name        = "allow-8080"
  description = "Allow access to port 8080"
}

resource "openstack_networking_secgroup_v2" "allow_8081" {
  name        = "allow-8081"
  description = "Allow access to port 8081"
}

resource "openstack_networking_secgroup_v2" "allow_http_80" {
  name        = "allow-http-80"
  description = "Allow HTTP access on port 80"
}

resource "openstack_networking_secgroup_v2" "allow_9090" {
  name        = "allow-9090"
  description = "Allow access to port 9090"
}

# Define Networking Resources

resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-mlops-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-mlops-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet2_ports" {
  for_each   = var.nodes
  name       = "sharednet2-${each.key}-mlops-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet2.id
  security_group_ids = [
    openstack_networking_secgroup_v2.allow_ssh.id,
    openstack_networking_secgroup_v2.allow_9001.id,
    openstack_networking_secgroup_v2.allow_8000.id,
    openstack_networking_secgroup_v2.allow_8080.id,
    openstack_networking_secgroup_v2.allow_8081.id,
    openstack_networking_secgroup_v2.allow_http_80.id,
    openstack_networking_secgroup_v2.allow_9090.id
  ]
}

# Define Compute Instances

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name        = "${each.key}-mlops-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.medium"
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.sharednet2_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

# Floating IP Assignment

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet2_ports["node1"].id
}
