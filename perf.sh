
for i in $(seq 2 64);
do

cmake .. -DDEEP_TREE=OFF -DNUM_BINS=$i > /dev/null 2> /dev/null
make -j > /dev/null 2> /dev/null

echo "\naneurism, Bins: $i, shallow  " >> results.txt
echo   "========================\n" >> results.txt
echo "\naneurism, Bins: $i, shallow  " >> details.txt
echo   "========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview aneurism.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nbonsai, Bins: $i, shallow  " >> results.txt
echo   "======================\n" >> results.txt
echo "\nbonsai, Bins: $i, shallow  " >> details.txt
echo   "======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview bonsai_tfnice.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nxmas, Bins: $i, shallow  " >> results.txt
echo   "====================\n" >> results.txt
echo "\nxmas, Bins: $i, shallow  " >> details.txt
echo   "====================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview xmastree512.xvf -p -camera xmas-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nstag, Bins: $i, shallow  " >> results.txt
echo   "====================\n" >> results.txt
echo "\nstag, Bins: $i, shallow  " >> details.txt
echo   "====================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview stagbeetle1024.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg256, Bins: $i, shallow\n" >> results.txt
echo "\n=======================\n" >> results.txt
echo "\ntvcg256, Bins: $i, shallow\n" >> details.txt
echo "\n=======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_256_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg512, Bins: $i, shallow\n" >> results.txt
echo "\n=======================\n" >> results.txt
echo "\ntvcg512, Bins: $i, shallow\n" >> details.txt
echo "\n=======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_512_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg1024, Bins: $i, shallow\n" >> results.txt
echo "\n========================\n" >> results.txt
echo "\ntvcg1024, Bins: $i, shallow\n" >> details.txt
echo "\n========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_1024_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg2048, Bins: $i, shallow\n" >> results.txt
echo "\n========================\n" >> results.txt
echo "\ntvcg2048, Bins: $i, shallow\n" >> details.txt
echo "\n========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview tvcg_2048_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

done


for i in $(seq 2 64);
do

cmake .. -DDEEP_TREE=ON -DNUM_BINS=$i > /dev/null 2> /dev/null
make -j > /dev/null 2> /dev/null

echo "\naneurism, Bins: $i, deep  " >> results.txt
echo   "========================\n" >> results.txt
echo "\naneurism, Bins: $i, deep  " >> details.txt
echo   "========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview aneurism.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nbonsai, Bins: $i, deep  " >> results.txt
echo   "======================\n" >> results.txt
echo "\nbonsai, Bins: $i, deep  " >> details.txt
echo   "======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview bonsai_tfnice.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nxmas, Bins: $i, deep  " >> results.txt
echo   "====================\n" >> results.txt
echo "\nxmas, Bins: $i, deep  " >> details.txt
echo   "====================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview xmastree512.xvf -p -camera xmas-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\nstag, Bins: $i, deep  " >> results.txt
echo   "====================\n" >> results.txt
echo "\nstag, Bins: $i, deep  " >> details.txt
echo   "====================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview stagbeetle1024.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg256, Bins: $i, deep\n" >> results.txt
echo "\n=======================\n" >> results.txt
echo "\ntvcg256, Bins: $i, deep\n" >> details.txt
echo "\n=======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_256_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg512, Bins: $i, deep\n" >> results.txt
echo "\n=======================\n" >> results.txt
echo "\ntvcg512, Bins: $i, deep\n" >> details.txt
echo "\n=======================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_512_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg1024, Bins: $i, deep\n" >> results.txt
echo "\n========================\n" >> results.txt
echo "\ntvcg1024, Bins: $i, deep\n" >> details.txt
echo "\n========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview ../build/tvcg_1024_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

echo "\ntvcg2048, Bins: $i, deep\n" >> results.txt
echo "\n========================\n" >> results.txt
echo "\ntvcg2048, Bins: $i, deep\n" >> details.txt
echo "\n========================\n" >> details.txt
VV_RENDERER=rayrendsimple vglrun RelWithDebInfo/bin/vview tvcg_2048_tf1.xvf -p -camera virvo-camera.txt -size 2160 2160 -benchmark >> results.txt 2>> details.txt

done
