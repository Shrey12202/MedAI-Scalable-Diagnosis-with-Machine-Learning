variable "suffix" {
  description = "Suffix for all resource names"
  type        = string
  default     = "project44"
}

variable "key_name" {
  description = "SSH keypair name"
  type        = string
  default     = "id_rsa_chameleon_project44"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}
