require 'Provider'
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)