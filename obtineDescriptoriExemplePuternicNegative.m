function descriptoriExemplePuternicNegative = obtineDescriptoriExemplePuternicNegative(parametri)
% descriptoriExempleNegative = matrice MxD, unde:
%   M = numarul de exemple puternic negative de antrenare (NU sunt fete de oameni),
%   M = parametri.numarExemplePuternicNegative
%   D = numarul de dimensiuni al descriptorului
%   in mod implicit D = (parametri.dimensiuneFereastra/parametri.dimensiuneCelula)^2*parametri.dimensiuneDescriptorCelula

imgFiles = dir( fullfile( parametri.numeDirectorExemplePuternicNegative, '*.jpg') ); %exemplele puternic negative sunt stocate ca .jpg
numarImagini = length(imgFiles);
%daca facem flip la imagine atunci putem obtine de 2 ori mai multe imagini
%pentru antrenare
descriptoriExemplePuternicNegative = zeros(numarImagini*2,(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2*parametri.dimensiuneDescriptorCelula);
disp(['Exista un numar de exemple pozitive = ' num2str(numarImagini)]);pause(2);
adaugati = 0;
for idx = 1:numarImagini
    disp(['Procesam exemplul pozitiv numarul ' num2str(idx)]);
    img = imread([parametri.numeDirectorExemplePuternicNegative '/' imgFiles(idx).name]);
    if size(img,3) > 1
        img = rgb2gray(img);
    end   
    %completati codul functiei in continuare
    adaugati = adaugati + 1;
    descriptorHOGImagine = vl_hog(single(img),parametri.dimensiuneCelulaHOG);
    descriptoriExemplePuternicNegative(adaugati,:) = descriptorHOGImagine(:)';
    
    adaugati = adaugati + 1;
    img_aux = flip(img,2);
    descriptorHOGImagine = vl_hog(single(img_aux),parametri.dimensiuneCelulaHOG);
    descriptoriExemplePuternicNegative(adaugati,:) = descriptorHOGImagine(:)';
    
    
end