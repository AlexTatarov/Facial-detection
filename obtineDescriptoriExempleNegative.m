function descriptoriExempleNegative = obtineDescriptoriExempleNegative(parametri)
% descriptoriExempleNegative = matrice MxD, unde:
%   M = numarul de exemple negative de antrenare (NU sunt fete de oameni),
%   M = parametri.numarExempleNegative
%   D = numarul de dimensiuni al descriptorului
%   in mod implicit D = (parametri.dimensiuneFereastra/parametri.dimensiuneCelula)^2*parametri.dimensiuneDescriptorCelula

imgFiles = dir( fullfile( parametri.numeDirectorExempleNegative , '*.jpg' ));
numarImagini = length(imgFiles);

numarExempleNegative_pe_imagine = round(parametri.numarExempleNegative/numarImagini);
descriptoriExempleNegative = zeros(parametri.numarExempleNegative,(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG)^2*parametri.dimensiuneDescriptorCelula);
disp(['Exista un numar de imagini = ' num2str(numarImagini) ' ce contine numai exemple negative']);
adaugati = 0;
for idx = 1:numarImagini
    disp(['Procesam imaginea numarul ' num2str(idx)]);
    img = imread([parametri.numeDirectorExempleNegative '/' imgFiles(idx).name]);
    if size(img,3) == 3
        img = rgb2gray(img);
    end 
    
    height = size(img,1);
    width = size(img,2);
    val_aux = 36;
    %val_aux = parametri.dimensiuneFerestra;
    %completati codul functiei in continuare
    
    for i = 1:numarExempleNegative_pe_imagine
        y = randi(height-val_aux+1);
        x = randi(width-val_aux+1);
        descriptorHOGImagine = vl_hog(single(img(y:y+val_aux-1,x:x+val_aux-1)),parametri.dimensiuneCelulaHOG);
        adaugati = adaugati + 1;
        descriptoriExempleNegative(adaugati,:) = descriptorHOGImagine(:)';
        
    end
    
end