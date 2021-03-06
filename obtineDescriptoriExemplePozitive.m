function descriptoriExemplePozitive = obtineDescriptoriExemplePozitive(parametri)
% descriptoriExemplePozitive = matrice NxD, unde:
%   N = numarul de exemple pozitive de antrenare (fete de oameni) 
%   D = numarul de dimensiuni al descriptorului
%   in mod implicit D = (parametri.dimensiuneFereastra/parametri.dimensiuneCelula)^2*parametri.dimensiuneDescriptorCelula

imgFiles = dir( fullfile( parametri.numeDirectorExemplePozitive, '*.jpg') ); %exemplele pozitive sunt stocate ca .jpg
numarImagini = length(imgFiles);
%daca facem flip la imagine atunci putem obtine de 2 ori mai multe imagini
%pentru antrenare
descriptoriExemplePozitive = zeros(numarImagini*2,(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2*parametri.dimensiuneDescriptorCelula);
disp(['Exista un numar de exemple pozitive = ' num2str(numarImagini)]);pause(2);
adaugati = 0;
for idx = 1:numarImagini
    disp(['Procesam exemplul pozitiv numarul ' num2str(idx)]);
    img = imread([parametri.numeDirectorExemplePozitive '/' imgFiles(idx).name]);
    if size(img,3) > 1
        img = rgb2gray(img);
    end   
    %completati codul functiei in continuare
    adaugati = adaugati + 1;
    descriptorHOGImagine = vl_hog(single(img),parametri.dimensiuneCelulaHOG);
    descriptoriExemplePozitive(adaugati,:) = descriptorHOGImagine(:)';
    
    adaugati = adaugati + 1;
    img_aux = flip(img,2);
    descriptorHOGImagine = vl_hog(single(img_aux),parametri.dimensiuneCelulaHOG);
    descriptoriExemplePozitive(adaugati,:) = descriptorHOGImagine(:)';
    
    
end