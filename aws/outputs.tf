output "cuda_ip" {
  value = aws_spot_instance_request.cuda.public_dns
  description = "Spot instance IP"
}