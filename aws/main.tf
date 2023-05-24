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
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDV9brHJCm7kD7yJ56DEjhU7sckMLa/wj3tWGuyhHjma0grspV6fyAgPaBzZq1RX0sujlgdIBR4b1i9fIUrcp4OcdZLaxdupJbUvew8BqwWgJnmjuwSvhrrHPdweJK5LIE82SYakM0ptBWpRzaRLGoz9P71ElVRIPtVYDAbHSrHrxy7jX5H+7ExTkKcUDYeGXaeAlzxNM6qDaoPI9APX2MbR/L6HzrwbfiUb6U8jLqJOYghCwwl8A6DjaTBBGqDMDcT5yjMF0Y39hNTfrYzxQ6V6Uq6+WEHLnE+WLUlblvwJtMhfsQvmFQybROsGY5clUa+6pW7V7TAHL8RuG7LufmpvgQK0ci2ummeIYGk6IZQHAl+9HFgnHDqe/ZpTFewnfeb2kpZIweogWpzouAjIuWmRXn2kuKggy1p45BbPnXOhyFGZlkJhoHO0iqrbWL5N8RWZ1JjviJrQo2QN5Iqo6q7hzbeCxAH2Ksvn+bgnZCGtBfjXejfOk+vKXxlBHMJh58= dawids@dawid-ms7d43"
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
              printf 'export PATH="/usr/local/cuda/bin:$PATH"\nexport LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n' >> /home/ec2-user/.bashrc
              EOF
}
