function cae_gui
% CAE_GUI An app for taking care of cross age effect tests
% that would be done to the participants. It shows images
% and records the age-guess accurracy of the participants.

clc
close all
%% Show image and get answer
trialNo = 1; % num of images
qq = 1;
cont = true;
% f = figure('Position',[360,500,450,285]);
imgDir = dir('/Volumes/Storage/MATLAB/crossageeffect/images');
aux = struct2cell(imgDir);
imgIndex = aux(1,3:length(aux));
agesGuess = zeros(1,trialNo);
agesReal = zeros(1,trialNo);
imgNames = strings(1,trialNo);
fN = inputdlg('Please enter your first name');
fName = fN{1};
sN = inputdlg('Please enter your surname');
sName = sN{1};
ageN = inputdlg('Please enter your age');
agePart = ageN{1};
while cont
    figure
    idx = randi(length(imgIndex),1);
    img_str = ['images/', imgIndex{idx}];
    if contains(img_str,'._') || contains(img_str,'_.')
        img_str = ['images/',extractAfter(img_str,9)];
    end
    auxval = extractBetween(img_str,12,13);
    agesReal(qq) = str2num(auxval{1});
    img = imread(img_str);
    imgNames(qq) = img_str;
    imshow(img);
    pause(5) % time to show image
    close 1
%     figure
    ageGuess = inputdlg('Whats the age?');
    if ~isempty(ageGuess)
        agesGuess(qq) = str2num(ageGuess{1});
        qq = qq + 1;
    end
    if qq > trialNo
        cont = false;
    end
end
%% Write to files
dt = date;
dtt = erase(datestr(datetime),' ');
outText = strjoin(['First Name: ', fName, ' Surname: ', sName, ' Age of Participant: ', agePart, '\n', 'Date & Time: ', dtt, '\n','Guessed ages: ', num2str(agesGuess), ' Real ages: ', num2str(agesReal), '\n', 'Images: ', imgNames]);
fileTxt = ['results/', [fName,sName],'/', fName, '_', sName, '_', dtt, '.txt'];
mkdir(['results/', fName, sName])
fid = fopen(fileTxt,'w');
fprintf(fid, outText);
fclose(fid);
agesOutput.defn = ["Participant age"; "Guess"; "Real"];
agesMat = str2num(agePart)*ones(1, trialNo);
agesOutput.vals = [agesMat; agesGuess; agesReal];
outStr = ['results/', fName, sName, '/', fName, sName, dtt, '_results.mat'];
save(outStr, 'agesOutput');
%%
figure
plot(agesGuess,'bx')
hold on
plot(agesReal,'ro')
grid on
axis auto
title('Guessed ages vs Real ages')
legend('Guessed ','Real')
end