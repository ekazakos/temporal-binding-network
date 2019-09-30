
echo 'Downloading pretrained TSN weights for Flow on Kinetics...'
echo
echo

wget "https://docs.google.com/uc?export=download&id=1vO_vALz8HXifZaEgtLBvtEq869AKf6kc" -O "kinetics_tsn_flow.pth.tar" 

echo 'Downloading pretrained TBN weights for RGB-Flow-Audio on EPIC-Kitchens...'
echo 
echo

query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1c2z0xrshfpLvhcbkIpNJVcdyPe5rEO-g" | pup 'a#uc-download-link attr{href}' | sed -e 's/amp;//g'`

curl -b ./cookie.txt -L -o "epic_tbn_rgbflowaudio.pth.tar" "https://drive.google.com${query}"

rm "cookie.txt"