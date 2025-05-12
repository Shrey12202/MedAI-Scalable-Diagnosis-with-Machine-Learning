import subprocess
import os

TERRAFORM_DIR = os.path.abspath("infra/provision/terraform")
ANSIBLE_DIR = os.path.abspath("infra/configure/ansible")
INVENTORY_FILE = os.path.join(ANSIBLE_DIR, "inventory.ini")
PLAYBOOK_FILE = os.path.join(ANSIBLE_DIR, "site.yml")

def run_terraform():
    subprocess.run(["terraform", "init"], cwd=TERRAFORM_DIR, check=True)
    subprocess.run(["terraform", "apply", "-auto-approve"], cwd=TERRAFORM_DIR, check=True)

def run_ansible():
    subprocess.run([
        "ansible-playbook", "-i", INVENTORY_FILE, PLAYBOOK_FILE
    ], cwd=ANSIBLE_DIR, check=True)

if __name__ == "__main__":
    run_terraform()
    run_ansible()
