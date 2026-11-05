import sys, torch, timm, hashlib
sys.path.insert(0, "src/ssl4polyp/models/mae")
import models_mae
torch.manual_seed(0)
m = models_mae.mae_vit_base_patch16(norm_pix_loss=False)
names=[n for n,_ in m.named_parameters()]
init=torch.cat([p.detach().flatten() for p in m.parameters()])
g=torch.Generator().manual_seed(123); x=torch.randn(2,3,224,224,generator=g)
torch.manual_seed(7)
loss,_,_=m(x,mask_ratio=0.75); loss.backward()
grads=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in m.parameters()])
wd=[n for n,p in m.named_parameters() if p.ndim==1 or n.endswith(".bias") or n in ("pos_embed","cls_token","mask_token","decoder_pos_embed")]
torch.save({"names":names,"init":init,"loss":loss.detach(),"grads":grads},sys.argv[1])
print(timm.__version__, torch.__version__, "params",init.numel(),"loss %.15g"%loss.item())
