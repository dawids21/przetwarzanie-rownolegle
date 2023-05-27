resource "aws_security_group" "cuda" {
  name = "cuda"
  ingress {
    from_port   = 22
    protocol    = "tcp"
    to_port     = 22
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    protocol    = "tcp"
    to_port     = 65535
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "cuda" {
  owners      = ["amazon"]
  most_recent = true
  filter {
    name   = "name"
    values = ["Deep Learning Base AMI (Amazon Linux 2)*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_key_pair" "cuda" {
  key_name   = "cuda"
  public_key = var.public_key
}

resource "aws_spot_instance_request" "cuda" {
  ami                         = data.aws_ami.cuda.id
  instance_type               = "g3s.xlarge"
  wait_for_fulfillment        = true
  vpc_security_group_ids      = [aws_security_group.cuda.id]
  associate_public_ip_address = true
  key_name                    = aws_key_pair.cuda.key_name
  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              printf 'export PATH="/usr/local/cuda/bin:$PATH"\nexport LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"\n' >> /home/ec2-user/.bashrc
              EOF
}
