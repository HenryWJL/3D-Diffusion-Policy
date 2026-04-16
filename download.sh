policy=${1}
task=${2}

cd checkpoints

wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D600_seed%3D0.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1200_seed%3D0.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1800_seed%3D0.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D2400_seed%3D0.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D3000_seed%3D0.pth

wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D600_seed%3D1.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1200_seed%3D1.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1800_seed%3D1.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D2400_seed%3D1.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D3000_seed%3D1.pth

wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D600_seed%3D2.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1200_seed%3D2.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D1800_seed%3D2.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D2400_seed%3D2.pth
wget -c https://huggingface.co/HenryWJL/${policy}/resolve/main/${task}/epoch%3D3000_seed%3D2.pth

mv epoch=600_seed=0.pth epoch%3D600_seed%3D0.pth
mv epoch=1200_seed=0.pth epoch%3D1200_seed%3D0.pth
mv epoch=1800_seed=0.pth epoch%3D1800_seed%3D0.pth
mv epoch=2400_seed=0.pth epoch%3D2400_seed%3D0.pth
mv epoch=3000_seed=0.pth epoch%3D3000_seed%3D0.pth

mv epoch=600_seed=1.pth epoch%3D600_seed%3D1.pth
mv epoch=1200_seed=1.pth epoch%3D1200_seed%3D1.pth
mv epoch=1800_seed=1.pth epoch%3D1800_seed%3D1.pth
mv epoch=2400_seed=1.pth epoch%3D2400_seed%3D1.pth
mv epoch=3000_seed=1.pth epoch%3D3000_seed%3D1.pth

mv epoch=600_seed=2.pth epoch%3D600_seed%3D2.pth
mv epoch=1200_seed=2.pth epoch%3D1200_seed%3D2.pth
mv epoch=1800_seed=2.pth epoch%3D1800_seed%3D2.pth
mv epoch=2400_seed=2.pth epoch%3D2400_seed%3D2.pth
mv epoch=3000_seed=2.pth epoch%3D3000_seed%3D2.pth